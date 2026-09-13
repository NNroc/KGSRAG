import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import KGSRAG, QueryParam
from rag.llm import hf_model_complete, hf_embedding
from rag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import re
import argparse
import json
import asyncio
import time


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def extract_and_convert_response_pubmedqa(text):
    # 定义正则表达式模式，匹配 ["answer_long": "", "answer_decision": ""]
    pattern = r'"answer_long": "([^"]*)"'
    # 使用正则表达式查找所有匹配项
    answer_long_matches = re.findall(pattern, text)
    pattern = r'"answer_decision": "([^"]*)"'
    answer_decision_matches = re.findall(pattern, text)
    try:
        answer_long = answer_long_matches[0]
        answer_decision = answer_decision_matches[0]
    except Exception as e:
        print(e)
        return {"answer_long": "", "answer_decision": ""}
    return {"answer_long": answer_long, "answer_decision": answer_decision}


def extract_and_convert_response_bioasq(text, type):
    if type == "yesno":
        if "yes" in text and "no" in text:
            return "ERROR"
        elif "no" in text:
            return "no"
        elif "yes" in text:
            return "yes"
        else:
            return "ERROR"
    elif type == "fastoid":
        pattern = r'"([^"]*)"'
        answer = re.findall(pattern, text)
        return answer
    elif type == "list":
        pattern = r'"([^"]*)"'
        answer = re.findall(pattern, text)
        return answer
    else:
        return text


async def process_query(query_text, rag_instance, query_param, query_type=None):
    result = await rag_instance.aquery(query_text, param=query_param, query_type=query_type)
    return {"query": query_text, "result": result}, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmedqa', help='pubmedqa/bioasq')
    parser.add_argument('--ideal', action='store_true', help='long answer')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Specific model name or save path')
    parser.add_argument('--decomposition', type=str, default='statement',
                        help='query decomposition: question/statement')
    parser.add_argument('--mode', type=str, default='all',
                        help='query mode: naive/perfect/keyword/all')
    parser.add_argument('--dynamic_threshold', action='store_true', help='dynamic threshold: True/False')
    parser.add_argument('--statement_threshold', type=float, default=0.40, help='statement_threshold')
    parser.add_argument('--filter', action='store_true', help='retrieval filtering: True/False')
    parser.add_argument('--save_dir', type=str, default='./output', help='save path')
    parser.add_argument('--suffix', type=str, default='-2', help='-0/-1/-2')
    parser.add_argument('--eval_file', type=str, default="", help='')
    parser.add_argument('--demo', type=int, default=2000, help='use in test')
    parser.add_argument('--min_threshold', type=float, default=0.1, help='min_threshold for dynamic retrieval')
    parser.add_argument('--max_threshold', type=float, default=0.5, help='max_threshold for dynamic retrieval')
    parser.add_argument('--alpha', type=int, default=10, help='max number of keywords for retrieval')
    parser.add_argument('--beta', type=int, default=5, help='max number of descriptions to keep per entity/relation')
    parser.add_argument('--cache_suffix', type=str, default='', help='custom suffix for cache file, e.g. "hp_tmin02"')
    args = parser.parse_args()
    dataset = args.dataset
    data_qa = json.load(open(f'data/{dataset}_qa.json', 'r'))
    model_name = args.model_name.split('/')[-1]
    query_param = QueryParam(dataset=dataset, decomposition=args.decomposition, mode=args.mode, top_k=60,
                             dynamic_threshold=args.dynamic_threshold, statement_threshold=args.statement_threshold,
                             filter=args.filter, max_token_for_text_unit=4000,
                             min_threshold=args.min_threshold, max_threshold=args.max_threshold,
                             alpha=args.alpha, beta=args.beta)
    print(query_param)
    WORKING_DIR = args.save_dir + '/' + dataset + "_" + model_name + args.suffix
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    rag = KGSRAG(
        working_dir=WORKING_DIR,
        dataset=dataset,
        llm_model_func=hf_model_complete,
        llm_model_name=args.model_name,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=lambda texts: hf_embedding(
                texts,
                tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
                embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to("cuda"),
            ),
        ),
    )

    if args.cache_suffix:
        from rag.utils import load_json
        new_cache_file = os.path.join(WORKING_DIR, f"kv_store_llm_response_cache_{args.cache_suffix}.json")
        rag.llm_response_cache._file_name = new_cache_file
        rag.llm_response_cache._data = load_json(new_cache_file) or {}
        print(f"Cache file: {new_cache_file}")

    loop = always_get_an_event_loop()
    if dataset == "pubmedqa":
        queries = [document["question"] for document in data_qa]
        answer_long = [document["answer_long"] for document in data_qa]
        answer_decision = [document["answer_decision"] for document in data_qa]
        if args.demo < 1000:
            queries = queries[:args.demo]
            answer_long = answer_long[:args.demo]
            answer_decision = answer_decision[:args.demo]

        answer_list = []
        preds_all = {}
        preds_answer_decision = []
        preds_answer_long = []
        tokens = 0

        for query_text in tqdm(queries, desc="Processing queries", unit="query"):
            if len(args.eval_file) > 0:
                if args.mode not in ['naive', 'perfect', 'keyword', 'all']:
                    WORKING_DIR = '.'
                with open(WORKING_DIR + '/' + args.eval_file, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                data = data[args.mode]
                for d in data:
                    if data[d]['original_prompt'] == query_text:
                        extract_answer = extract_and_convert_response_pubmedqa(data[d]["return"])
                        break
            else:
                result, error = loop.run_until_complete(process_query(query_text, rag, query_param))
                if isinstance(result["result"], int):
                    tokens += result["result"]
                    continue
                extract_answer = extract_and_convert_response_pubmedqa(result["result"])

            preds_all[query_text] = {
                "answer_long": extract_answer["answer_long"],
                "answer_decision": extract_answer["answer_decision"]
            }

        if tokens > 0:
            print("Total Tokens:", tokens)
        else:
            for question in queries:
                preds_answer_decision.append(preds_all[question]["answer_decision"])
                preds_answer_long.append(preds_all[question]["answer_long"])

            acc = accuracy_score(answer_decision, preds_answer_decision)
            maf = f1_score(answer_decision, preds_answer_decision, average='macro')

            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            print(acc)
            print(maf)
    elif dataset == "bioasq":
        queries, ideal_answer, exact_answer, query_type = [], [], [], []
        for document in data_qa:
            if args.ideal:
                document["type"] = "ideal"
            else:
                if document["type"] == "summary":
                    continue
            queries.append(document["question"])
            ideal_answer.append(document["ideal_answer"])
            exact_answer.append(document["exact_answer"])
            query_type.append(document["type"])
        if args.demo < 1000:
            queries = queries[:args.demo]
            ideal_answer = ideal_answer[:args.demo]
            exact_answer = exact_answer[:args.demo]
            type = type[:args.demo]

        answer_list = []
        preds_all = {}
        preds_ideal_answer = []
        preds_exact_answer = []

        tokens = 0
        for idx, query_text in enumerate(tqdm(queries, desc="Processing queries", unit="query")):
            if query_text != "What proportion of alteration in NTRK genes are attributable to colorectal cancer?":
                continue
            result, error = loop.run_until_complete(process_query(query_text, rag, query_param, query_type[idx]))
            if isinstance(result["result"], int):
                tokens = tokens + result["result"]
        if tokens > 0:
            print("Total Tokens:", tokens)

        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    else:
        questions = ["who?"]
        result, error = loop.run_until_complete(process_query(questions, rag, query_param))
        print(result)
