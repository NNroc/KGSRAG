import re
import json

# 打开文件并逐行读取
with open('data/jty/军特药1-500.txt', 'r', encoding='utf-8') as file1, \
        open('data/jty/军特药501-1000.txt', 'r', encoding='utf-8') as file2, \
        open('data/jty/军特药1001-1026.txt', 'r', encoding='utf-8') as file3:
    lines1 = file1.readlines()
    lines2 = file2.readlines()
    lines3 = file3.readlines()
lines = lines1 + lines2 + lines3
text_data = ""
# 打印每一行
for line in lines:
    text_data = text_data + line

# 定义正则表达式来匹配题目和摘要
pattern = re.compile(r"(TI\s+.*?)(?=\nSO|\nLA).*?(AB\s+.*?)(?=\nC1|\nID)", re.DOTALL)

# 提取题目和摘要
results = pattern.findall(text_data)
text_list = []
for r in results:
    ti, ab = "", ""
    if r[0][:3] == "TI ":
        ti = r[0][3:]
        ti = ti.replace("\n", " ")
        ti = re.sub(r"\s+", " ", ti)
    if r[1][:3] == "AB ":
        ab = r[1][3:]
        ab = ab.replace("\n", " ")
        ab = re.sub(r"\s+", " ", ab)
    text = ti + "\n" + ab
    text_list.append(text)

save_path = './data/'
with open(save_path + 'custom_corpus.json', 'w', encoding='utf-8') as json_file:
    json.dump(text_list, json_file, ensure_ascii=False, indent=4)
