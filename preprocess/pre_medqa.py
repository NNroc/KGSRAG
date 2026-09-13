import json
import os


def merge_txt_files(folder_path, output_file):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    output_list = []
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 遍历文件夹中的所有文件
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            # 检查是否为txt文件（而不是文件夹或其他类型文件）
            if os.path.isfile(file_path) and filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as infile:
                    output_list.append(infile.read())
        json.dump(output_list, outfile, ensure_ascii=False, indent=4)


# 定义要读取的文件路径
qa_train_path = './data/MedQA/questions/US/train.jsonl'
qa_dev_path = './data/MedQA/questions/US/dev.jsonl'
qa_test_path = './data/MedQA/questions/US/test.jsonl'
corpus_path = './data/MedQA/textbooks/en'
save_path = './data/'
dataset = 'medqa'

# format dataset
with open(qa_test_path, 'r', encoding='utf-8') as file:
    qa_list = []
    for line in file:
        data = json.loads(line.strip())
        supplements = ""
        for supplement in data['options']:
            supplements = supplements + supplement + ': ' + data['options'][supplement] + '\n'
        qa_list.append({'question': data['question'] + '\n\n' + supplements,
                        'answer': data['answer'],
                        'answer_options': data['answer_idx']})

with open(save_path + dataset + '_test.json', 'w', encoding='utf-8') as json_file:
    json.dump(qa_list, json_file, ensure_ascii=False, indent=4)

merge_txt_files(corpus_path, save_path + dataset + '_corpus.json')
