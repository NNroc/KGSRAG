import argparse
import json
import re
import datetime
import os
from tqdm import tqdm
from openai import OpenAI


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


def batch_eval(reference_answers_file, response_answers_file, output_file_path):
    queries, query_types, reference_answers, response_answers = [], {}, [], []
    max_word = 0

    with open(reference_answers_file, "r") as f:
        answers1 = json.load(f)
    for qa in answers1:
        queries.append(qa["question"])
        if 'bioasq' in reference_answers_file:
            query_types[qa["question"]] = qa['type']
        else:
            query_types[qa["question"]] = 'summary'
        answer_list = []
        use_words = 1000
        ans_flag = "ideal_answer"
        if "bioasq" in reference_answers_file:
            ans_flag = "ideal_answer"
        elif "pubmed" in reference_answers_file:
            ans_flag = "answer_long"
            qa[ans_flag] = [qa[ans_flag]]
        for i in qa[ans_flag]:
            answer_list.append(i)
            # 计算长度
            cleaned_text = re.sub(r'[^\w\s]', '', i)
            words = cleaned_text.split()
            use_words = min(use_words, len(words))
        max_word = max(max_word, use_words)

        reference_answers.append(answer_list)

    with open(response_answers_file, "r") as f:
        answers2 = list(json.load(f).values())[0]

    results = []
    results_questions = []
    if os.path.exists(output_file_path):
        with open(output_file_path, "r", encoding="utf-8") as file:
            results = json.load(file)
        for r in results:
            results_questions.append(r["Question"])

    for qu in queries:
        for an in answers2:
            if answers2[an]["original_prompt"] == qu:
                if "pubmed" in reference_answers_file:
                    text = extract_and_convert_response_pubmedqa(answers2[an]["return"])["answer_long"]
                    response_answers.append(text)
                    break
                response_answers.append(answers2[an]["return"])
                break

    assert len(queries) == len(reference_answers) == len(response_answers), \
        "error in number (queries, reference_answers or response_answers)."

    requests = []
    for i, (query, reference_answers, response_answer) in enumerate(zip(queries, reference_answers, response_answers)):
        sys_prompt = """---Role---
You are a biomedical expert tasked with evaluating the answer to the questions based on some reference answers and four criteria: **Information Recall**, **Information Precision**, **Information Repetition**, and **Readability**.
"""

        prompt = f"""You will evaluate the answer to the questions based on some reference answers and four criteria: **Information Recall**, **Information Precision**, **Information Repetition**, and **Readability**.

- **Information Recall**: All the necessary information is reported.
- **Information Precision**: No irrelevant information is reported.
- **Information Repetition**: The answer does not repeat the same information multiple times.
- **Readability**: The answer is easily readable and fluent.

An 1–5 scale will be used in all four criteria (1 for "very poor" and 5 for "excellent") and provide reasons.

Here is the question:
{query}

Here are some reference answers:

**Reference Answers:**
{reference_answers}

Here are the answer that need to be evaluated:

**Answer:**
{response_answer}

Evaluate answer using the four criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:
{{
    "Information Recall": {{
        "Score": "1–5",
        "Explanation": "[Provide explanation here]"
    }},
    "Information Precision": {{
        "Score": "1–5",
        "Explanation": "[Provide explanation here]"
    }},
    "Information Repetition": {{
        "Score": "1–5",
        "Explanation": "[Provide explanation here]"
    }},
    "Readability": {{
        "Score": "1–5",
        "Explanation": "[Provide explanation here]"
    }}
}}
"""

        requests.append({"query": query, "sys_prompt": sys_prompt, "prompt": prompt})

    client = OpenAI(api_key="our api key",
                    base_url="https://api.siliconflow.cn/v1")

    score_recall, score_precision, score_repetition, score_readability = 0, 0, 0, 0
    for i, request in enumerate(tqdm(requests)):
        try:
            if request["query"] in results_questions:
                continue
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[
                    {"role": "system", "content": request["sys_prompt"]},
                    {"role": "user", "content": request["prompt"]},
                ],
                response_format={'type': 'json_object'},
                stream=False
            )
            result = response.choices[0].message.content
            data = json.loads(result)
            results.append({"Question": request["query"],
                            "Sys_prompt": request["sys_prompt"],
                            "Prompt": request["prompt"],
                            "Information Recall Score": data["Information Recall"]["Score"],
                            "Information Recall Explanation": data["Information Recall"]["Explanation"],
                            "Information Precision Score": data["Information Precision"]["Score"],
                            "Information Precision Explanation": data["Information Precision"]["Explanation"],
                            "Information Repetition Score": data["Information Repetition"]["Score"],
                            "Information Repetition Explanation": data["Information Repetition"]["Explanation"],
                            "Readability Score": data["Readability"]["Score"],
                            "Readability Explanation": data["Readability"]["Explanation"]
                            })
            with open(output_file_path, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=4, ensure_ascii=False)
        except Exception as e:
            print("ERROR: Question", request["query"])
            print(e)

    for r in results:
        # print(r["Question"])
        score_recall += int(r["Information Recall Score"])
        score_precision += int(r["Information Precision Score"])
        score_repetition += int(r["Information Repetition Score"])
        score_readability += int(r["Readability Score"])

    print("score_recall", score_recall / len(results))
    print("score_precision", score_precision / len(results))
    print("score_repetition", score_repetition / len(results))
    print("score_readability", score_readability / len(results))
    print("score_avg", (score_recall + score_precision + score_repetition + score_readability) / 4 / len(results))

    score_recall, score_precision, score_repetition, score_readability, score_num = \
        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]

    if 'bioasq' in reference_answers_file:
        for r in results:
            idx = 3
            query_type = query_types[r["Question"]]
            if query_type == 'yesno':
                idx = 0
            elif query_type == 'factoid':
                idx = 1
            elif query_type == 'list':
                idx = 2
            score_num[idx] += 1
            score_recall[idx] += int(r["Information Recall Score"])
            score_precision[idx] += int(r["Information Precision Score"])
            score_repetition[idx] += int(r["Information Repetition Score"])
            score_readability[idx] += int(r["Readability Score"])
        score_recall = [x / score_num[i] for i, x in enumerate(score_recall)]
        score_precision = [x / score_num[i] for i, x in enumerate(score_precision)]
        score_repetition = [x / score_num[i] for i, x in enumerate(score_repetition)]
        score_readability = [x / score_num[i] for i, x in enumerate(score_readability)]
        score_avg = [(a + b + c + d) / 4 for i, (a, b, c, d) in
                     enumerate(zip(score_recall, score_precision, score_repetition, score_readability))]
        for i in range(4):
            print(score_recall[i], score_precision[i], score_repetition[i], score_readability[i], score_avg[i])

    with open(output_file_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_file', type=str,
                        default='./data/bioasq_qa.json', help='Question answer filepath. bioasq_qa/pubmedqa_qa')
    parser.add_argument('--ga_file', type=str,
                        default='./output/bioasq_Qwen2.5-7B-Instruct-2/kv_store_llm_response_cache_naive_ideal_1.json',
                        help="Qenerate answer filepath.")
    args = parser.parse_args()

    output_file = "./output/evaluation/" + args.ga_file.replace("./output/", "").replace("/", "-")
    batch_eval(args.qa_file, args.ga_file, output_file)
    print(datetime.datetime.today())
