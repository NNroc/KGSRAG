import json

save_path = './data/'
dataset = 'bioasq'
hipporag_qa = []
hipporag_corpus = []
# 读取 qa.json 文件
with open(save_path + dataset + '_qa.json', 'r', encoding='utf-8') as json_file:
    qa_list = json.load(json_file)

# 读取 corpus.json 文件
with open(save_path + dataset + '_corpus.json', 'r', encoding='utf-8') as json_file:
    corpus_list = json.load(json_file)

for qa in qa_list:
    hipporag_qa.append({
        'id': qa['id'],
        'question': qa['question'],
        'answer': qa['exact_answer'],
        'long_answer': qa['ideal_answer'],
        'type': qa['type'],
        'supporting_facts': []
    })

for idx, corpus in enumerate(corpus_list):
    hipporag_corpus.append({
        'title': '',
        'text': corpus
    })

with open(save_path + 'hipporag/' + dataset + '.json', 'w', encoding='utf-8') as json_file:
    json.dump(hipporag_qa, json_file, ensure_ascii=False, indent=4)

with open(save_path + 'hipporag/' + dataset + '_corpus.json', 'w', encoding='utf-8') as json_file:
    json.dump(hipporag_corpus, json_file, ensure_ascii=False, indent=4)

save_path = './data/'
dataset = 'pubmedqa'
hipporag_qa = []
hipporag_corpus = []
# 读取 qa.json 文件
with open(save_path + dataset + '_qa.json', 'r', encoding='utf-8') as json_file:
    qa_list = json.load(json_file)

# 读取 corpus.json 文件
with open(save_path + dataset + '_corpus.json', 'r', encoding='utf-8') as json_file:
    corpus_list = json.load(json_file)

for idx, qa in enumerate(qa_list):
    hipporag_qa.append({
        'id': idx,
        'question': qa['question'],
        'answer': qa['answer_decision'],
        'long_answer': qa['answer_long'],
        'supporting_facts': []
    })

for idx, corpus in enumerate(corpus_list):
    hipporag_corpus.append({
        'title': '',
        'text': corpus
    })

with open(save_path + 'hipporag/' + dataset + '.json', 'w', encoding='utf-8') as json_file:
    json.dump(hipporag_qa, json_file, ensure_ascii=False, indent=4)

with open(save_path + 'hipporag/' + dataset + '_corpus.json', 'w', encoding='utf-8') as json_file:
    json.dump(hipporag_corpus, json_file, ensure_ascii=False, indent=4)
