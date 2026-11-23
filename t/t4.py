import os
import sys
import re
import argparse
import json
import asyncio
import time

data_qa = json.load(open(f'data/pubmedqa_qa.json', 'r'))

number, number_yes, number_no, number_maybe = 0, 0, 0, 0
question_length, question_length_yes, question_length_no, question_length_maybe = 0, 0, 0, 0
long_answer_length, long_answer_length_yes, long_answer_length_no, long_answer_length_maybe = 0, 0, 0, 0
retrieval_num, retrieval_num_yes, retrieval_num_no, retrieval_num_maybe = 0, 0, 0, 0

for document in data_qa:
    if document["answer_decision"] == "yes":
        number_yes += 1
        question_length_yes = question_length_yes + len(document["question"].split())
        long_answer_length_yes = long_answer_length_yes + len(document["answer_long"].split())
        retrieval_num_yes += len(document["contexts"])
    elif document["answer_decision"] == "no":
        number_no += 1
        question_length_no = question_length_no + len(document["question"].split())
        long_answer_length_no = long_answer_length_no + len(document["answer_long"].split())
        retrieval_num_no += len(document["contexts"])
    else:
        number_maybe += 1
        question_length_maybe = question_length_maybe + len(document["question"].split())
        long_answer_length_maybe = long_answer_length_maybe + len(document["answer_long"].split())
        retrieval_num_maybe += len(document["contexts"])
    number += 1
    question_length = question_length + len(document["question"].split())
    long_answer_length = long_answer_length + len(document["answer_long"].split())
    retrieval_num += len(document["contexts"])

print(question_length_yes / number_yes, long_answer_length_yes / number_yes, retrieval_num_yes / number_yes)
print(question_length_no / number_no, long_answer_length_no / number_no, retrieval_num_no / number_no)
print(question_length_maybe / number_maybe, long_answer_length_maybe / number_maybe,
      retrieval_num_maybe / number_maybe)
print(question_length / number, long_answer_length / number, retrieval_num / number)

data_qa = json.load(open(f'data/bioasq_qa.json', 'r'))

number, number_yes, number_no, number_factoid, number_list, number_summary = 0, 0, 0, 0, 0, 0
question_length, question_length_yes, question_length_no, question_length_factoid, question_length_list, question_length_summary = 0, 0, 0, 0, 0, 0
long_answer_length, long_answer_length_yes, long_answer_length_no, long_answer_length_factoid, long_answer_length_list, long_answer_length_summary = 0, 0, 0, 0, 0, 0
retrieval_num, retrieval_num_yes, retrieval_num_no, retrieval_num_factoid, retrieval_num_list, retrieval_num_summary = 0, 0, 0, 0, 0, 0

for document in data_qa:
    if document["type"] == "factoid":
        number_factoid += 1
        question_length_factoid = question_length_factoid + len(document["question"].split())
        num, ideal_answer_length = 0, 0
        for ideal_answer in document["ideal_answer"]:
            num += 1
            ideal_answer_length = ideal_answer_length + len(ideal_answer.split())
        long_answer_length_factoid = long_answer_length_factoid + ideal_answer_length / num
        retrieval_num_factoid += len(document["contexts"])
    elif document["type"] == "list":
        number_list += 1
        question_length_list = question_length_list + len(document["question"].split())
        num, ideal_answer_length = 0, 0
        for ideal_answer in document["ideal_answer"]:
            num += 1
            ideal_answer_length = ideal_answer_length + len(ideal_answer.split())
        long_answer_length_list = long_answer_length_list + ideal_answer_length / num
        retrieval_num_list += len(document["contexts"])
    elif document["type"] == "summary":
        number_summary += 1
        question_length_summary = question_length_summary + len(document["question"].split())
        num, ideal_answer_length = 0, 0
        for ideal_answer in document["ideal_answer"]:
            num += 1
            ideal_answer_length = ideal_answer_length + len(ideal_answer.split())
        long_answer_length_summary = long_answer_length_summary + ideal_answer_length / num
        retrieval_num_summary += len(document["contexts"])
    elif document["exact_answer"] == "yes":
        number_yes += 1
        question_length_yes = question_length_yes + len(document["question"].split())
        num, ideal_answer_length = 0, 0
        for ideal_answer in document["ideal_answer"]:
            num += 1
            ideal_answer_length = ideal_answer_length + len(ideal_answer.split())
        long_answer_length_yes = long_answer_length_yes + ideal_answer_length / num
        retrieval_num_yes += len(document["contexts"])
    elif document["exact_answer"] == "no":
        number_no += 1
        question_length_no = question_length_no + len(document["question"].split())
        num, ideal_answer_length = 0, 0
        for ideal_answer in document["ideal_answer"]:
            num += 1
            ideal_answer_length = ideal_answer_length + len(ideal_answer.split())
        long_answer_length_no = long_answer_length_no + ideal_answer_length / num
        retrieval_num_no += len(document["contexts"])
    number += 1
    question_length = question_length + len(document["question"].split())
    num, ideal_answer_length = 0, 0
    for ideal_answer in document["ideal_answer"]:
        num += 1
        ideal_answer_length = ideal_answer_length + len(ideal_answer.split())
    long_answer_length = long_answer_length + ideal_answer_length / num
    retrieval_num += len(document["contexts"])

print(question_length_yes / number_yes, long_answer_length_yes / number_yes, retrieval_num_yes / number_yes)
print(question_length_no / number_no, long_answer_length_no / number_no, retrieval_num_no / number_no)
print(question_length_factoid / number_factoid, long_answer_length_factoid / number_factoid,
      retrieval_num_factoid / number_factoid)
print(question_length_list / number_list, long_answer_length_list / number_list, retrieval_num_list / number_list)
print(question_length_summary / number_summary, long_answer_length_summary / number_summary,
      retrieval_num_summary / number_summary)
print(question_length / number, long_answer_length / number, retrieval_num / number)
