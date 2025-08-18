import os

os.environ["TRANSFORMERS_CACHE"] = "/data/pretrained/cache"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import numpy as np
import random
import sys
import argparse
import json

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import KGSRAG, QueryParam
from rag.llm import hf_model_complete, hf_embedding
from rag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from tqdm.asyncio import tqdm as tqdm_async

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmedqa', help='medqa/pubmedqa/bioasq')
    parser.add_argument('--model_name', type=str, default='/data/pretrained/Qwen/Qwen2.5-7B-Instruct',
                        help='Specific model name or save path')
    parser.add_argument('--decomposition', type=str, default='statement', help='origin/question/statement')
    parser.add_argument('--mode', type=str, default='none', help='naive/graph/keywords/all/none')
    parser.add_argument('--save_dir', type=str, default='./output', help='save path')
    parser.add_argument('--suffix', type=str, default='-2', help='-0/-1/-2')
    args = parser.parse_args()
    dataset = args.dataset
    corpus_question = json.load(open(f'data/{dataset}_qa.json', 'r'))
    model_name = args.model_name.split('/')[-1]
    query_param = QueryParam(decomposition=args.decomposition, mode=args.mode)
    WORKING_DIR = args.save_dir + '/' + dataset + "_" + model_name + args.suffix
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    rag = KGSRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name=args.model_name,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=lambda texts: hf_embedding(
                texts,
                tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
                embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            ),
        ),
    )

    question_docs = [{"question": document["question"]} for document in corpus_question]
    for question_doc in tqdm_async(question_docs, desc="question documents"):
        rag.query(question_doc["question"], query_param)
