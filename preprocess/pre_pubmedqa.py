import json

# 定义要读取的文件路径
qa_label_path = './data/PubMedQA/ori_pqal.json'
save_path = './data/'
dataset = 'pubmedqa'
corpus_list = []

# format dataset
with open(qa_label_path, 'r', encoding='utf-8') as file:
    # 加载JSON数据
    data = json.load(file)
    qa_list = []
    for k, v in data.items():
        contexts = []
        for context in v['CONTEXTS']:
            contexts.append(context)
        corpus_list.extend(contexts)
        qa_list.append({
            'question': v['QUESTION'],
            'answer_long': v['LONG_ANSWER'],
            'answer_decision': v['final_decision'],
            'contexts': contexts
        })

with open(save_path + dataset + '_qa.json', 'w', encoding='utf-8') as json_file:
    json.dump(qa_list, json_file, ensure_ascii=False, indent=4)

with open(save_path + dataset + '_corpus.json', 'w', encoding='utf-8') as json_file:
    json.dump(corpus_list, json_file, ensure_ascii=False, indent=4)
