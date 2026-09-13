import json
import os


def merge_txt_files(folder_path, output_file):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 遍历文件夹中的所有文件
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            # 检查是否为txt文件（而不是文件夹或其他类型文件）
            if os.path.isfile(file_path) and filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as infile:
                    # 读取文件内容并写入输出文件
                    outfile.write(infile.read())
                    outfile.write('\n')  # 添加换行符以分隔文件内容


# 定义要读取的文件路径
qa_train_path = './data/MedMCQA/train.json'
qa_dev_path = './data/MedMCQA/dev.json'
qa_test_path = './data/MedMCQA/test.json'
# corpus_path = './data/MedMCQA/en'
save_path = './data/'
dataset = 'medmcqa'

# format dataset
with open(qa_test_path, 'r', encoding='utf-8') as file:
    qa_list = []
    for line in file:
        data = json.loads(line.strip())
        supplements = ""
        supplements = supplements + 'A: ' + data['options']['opa'] + '\n'
        supplements = supplements + 'B: ' + data['options']['opb'] + '\n'
        supplements = supplements + 'C: ' + data['options']['opc'] + '\n'
        supplements = supplements + 'D: ' + data['options']['opd'] + '\n'
        supplements = supplements + 'choice type: ' + data['options']['choice_type'] + '\n'

        qa_list.append({'question': data['question'],
                        'supplement': supplements,
                        'answer': data['answer'],
                        'answer_options': data['answer_idx']})

with open(save_path + dataset + '_test.json', 'w', encoding='utf-8') as json_file:
    json.dump(qa_list, json_file, ensure_ascii=False, indent=4)

# merge_txt_files(corpus_path, save_path + dataset + '_corpus.json')
