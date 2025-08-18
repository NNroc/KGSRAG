import os
import sys

os.environ["TRANSFORMERS_CACHE"] = "/data/pretrained/cache"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import argparse
import json
import time


def extract_and_convert_response_bioasq(text, type):
    if type == "yesno":
        if text[:2].lower() == 'no':
            return 'no'
        elif text[:3].lower() == 'yes':
            return 'yes'
        pattern = r"\b(yes|no)\b(?!\s+\w)"
        answer = set()
        matches = re.findall(pattern, text, re.IGNORECASE)
        normalized_matches = [answer.add(match.lower()) for match in matches]
        answer = list(answer)
        if len(answer) != 1:
            # print('errors: ' + text[:20])
            return "errors!!!"
        else:
            answer = answer[0].lower()
            return answer
    elif type == "factoid":
        pattern = '\[[^\]]*\]'
        answer = re.findall(pattern, text)
        if len(answer) == 0:
            return []
        answer = str(max(answer, key=len))
        pattern = r'"([^"]*)"'
        answer = re.findall(pattern, answer)
        return [[a for a in answer if a != "entities"]]
    elif type == "list":
        pattern = '\[[^\]]*\]'
        answer = re.findall(pattern, text)
        if len(answer) == 0:
            return []
        answer = str(max(answer, key=len))
        pattern = r'"([^"]*)"'
        answer = re.findall(pattern, answer)
        answers = []
        for a in answer:
            if a == "entities":
                continue
            answers.append([a])
        return answers
    else:
        print('errors!!!')
        return "errors!!!"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_model', type=str, default='Qwen2.5-7B-Instruct')
    parser.add_argument('--answer_filename', type=str, default='kv_store_llm_response_cache.json')
    parser.add_argument('--filepath', type=str, default='')
    parser.add_argument('--mode', type=str, default='naive', help='naive/keyword/statement/all')
    parser.add_argument('--suffix', type=str, default='-2', help='-0/-1/-2')
    args = parser.parse_args()

    question_file = './data/bioasq_qa.json'
    question_data = json.load(open(question_file, 'r'))
    if len(args.filepath) > 5:
        answer_data = json.load(open(args.filepath, 'r'))
    else:
        answer_file = './output/bioasq_' + args.answer_model + args.suffix + '/' + args.answer_filename
        answer_data = json.load(open(answer_file, 'r'))
    result_data = []
    answer_data = answer_data[args.mode]
    answer_result = {}
    errors_map = {}
    for an in answer_data:
        answer_result[answer_data[an]['original_prompt']] = answer_data[an]['return']

    for document in question_data:
        if document['type'] == 'summary':
            continue
        answer_unextract = answer_result[document['question']]
        answer = extract_and_convert_response_bioasq(answer_result[document['question']], document['type'])
        if answer == "errors!!!":
            errors_map[document['type']] = errors_map.get(document['type'], 0) + 1
        result_data.append({
            'id': document['id'],
            'body': document['question'],
            'type': document['type'],
            'output_answer': answer_result[document['question']],
            'ideal_answer': "",
            'exact_answer': answer
        })

    print(errors_map)
    response = {'questions': result_data}

    with open('./evaluate/flat/BioASQEvaluation/evaluation/system_response.json', 'w', encoding='utf-8') as json_file:
        json.dump(response, json_file, ensure_ascii=False, indent=4)
