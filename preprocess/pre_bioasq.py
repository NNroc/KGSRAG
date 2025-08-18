import json
from Bio import Entrez, Medline


def get_titles_and_abstracts(pmid_list):
    # 将PMID列表转换为逗号分隔的字符串
    pmid_string = ",".join(pmid_list)
    # 使用efetch获取文献记录
    handle = Entrez.efetch(db="pubmed", id=pmid_string, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    handle.close()

    contexts = []
    # 遍历每条记录并提取信息
    for record in records:
        contexts.append(record.get("TI") + '\n' + record.get("AB"))

    return contexts


Entrez.email = "your_email@example.com"
# 定义要读取的文件路径
qa_label_paths = ['./data/BioASQ-QA/12B1_golden.json', './data/BioASQ-QA/12B2_golden.json',
                  './data/BioASQ-QA/12B3_golden.json', './data/BioASQ-QA/12B4_golden.json']
save_path = './data/'
dataset = 'bioasq'
qa_list = []
corpus_list = []
for qa_label_path in qa_label_paths:
    print(qa_label_path)
    # format dataset
    with open(qa_label_path, 'r', encoding='utf-8') as file:
        # 加载JSON数据
        data = json.load(file)
        data = data['questions']
        for v in data:
            pmids = []
            for context in v['documents']:
                pmids.append(context.split('/')[-1])
            contexts = get_titles_and_abstracts(pmids)
            corpus_list.extend(contexts)
            qa_list.append({
                'id': v['id'],
                'question': v['body'],
                'ideal_answer': v.get('ideal_answer', ''),
                'exact_answer': v.get('exact_answer', ''),
                'type': v['type'],
                'pmids': pmids,
                'contexts': contexts
            })

with open(save_path + dataset + '_qa.json', 'w', encoding='utf-8') as json_file:
    json.dump(qa_list, json_file, ensure_ascii=False, indent=4)

with open(save_path + dataset + '_corpus.json', 'w', encoding='utf-8') as json_file:
    json.dump(corpus_list, json_file, ensure_ascii=False, indent=4)
