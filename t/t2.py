import re

data = """
0.803921568627451 0.3176470588235294 0.3764705882352941 0.3450980392156863 0.3832056444372621 0.29187852741621123 0.3157154326911636 0.7879417879417879 0.8461538461538461 0.7297297297297297
YesNo Acc: 0.803921568627451
Factoid Strict Acc: 0.3176470588235294
Factoid Lenient Acc: 0.3764705882352941
Factoid MRR: 0.3450980392156863
List Prec: 0.3832056444372621
List Rec: 0.29187852741621123
List F1: 0.3157154326911636
YesNo macroF1: 0.7879417879417879
YesNo F1 yes: 0.8461538461538461
YesNo F1 no: 0.7297297297297297
"""

# 按行分割数据
lines = data.strip().split("\n")

# 遍历每一行，将数值用制表符连接
for line in lines:
    numbers = line.split(' ')
    if len(numbers) < 9:
        continue
    results = []
    results.append(numbers[0])
    results.append(numbers[7])
    results.append(numbers[8])
    results.append(numbers[9])
    results.append(-1)
    results.append(numbers[1])
    results.append(numbers[2])
    results.append(numbers[3])
    results.append(numbers[4])
    results.append(numbers[5])
    results.append(numbers[6])
    print("\t".join(str(item) for item in results))
