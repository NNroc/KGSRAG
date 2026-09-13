"""
Traditional reference-based evaluation metrics (ROUGE-L, BERTScore)
for validating LLM-as-a-Judge results in long-form text evaluation.

Supports both bioasq and pubmedqa datasets.

Usage:
    # BioASQ
    python evaluate/ngram_eval.py \
        --dataset bioasq \
        --qa_file ./data/bioasq_qa.json \
        --answer_file ./output/bioasq_Qwen2.5-7B-Instruct-2/kv_store_llm_response_cache_all_ideal_6.json \
        --output_file ./output/evaluation/ngram_metrics_bioasq_Qwen2.5-7B-Instruct-2.json

    # PubMedQA
    python evaluate/ngram_eval.py \
        --dataset pubmedqa \
        --qa_file ./data/pubmedqa_qa.json \
        --answer_file ./output/pubmedqa_Qwen2.5-7B-Instruct-2/kv_store_llm_response_cache_all_1.json \
        --output_file ./output/evaluation/ngram_metrics_pubmedqa_Qwen2.5-7B-Instruct-2.json
"""

import os
import sys
import json
import argparse
from collections import defaultdict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rouge_score import rouge_scorer

# BERTScore is optional - may fail if model is not cached
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except Exception as e:
    print(f"Warning: BERTScore not available ({e}). Will skip BERTScore computation.")
    BERTSCORE_AVAILABLE = False


def parse_pubmedqa_answer(return_str):
    """Parse pubmedqa answer_long from return string using regex (similar to long_evaluate.py)."""
    import re
    # Use regex to extract answer_long, similar to extract_and_convert_response_pubmedqa
    pattern = r'"answer_long": "([^"]*)"'
    matches = re.findall(pattern, return_str)
    if matches:
        return matches[0]
    return ""


def load_answer_file(answer_file, dataset):
    """Load answer file and extract question-answer pairs."""
    with open(answer_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qa_pairs = []

    # Find the mode key (all, naive, keyword, etc.)
    mode_key = None
    for key in data.keys():
        if key in ['all', 'naive', 'keyword']:
            mode_key = key
            break

    if mode_key is None:
        # Try first key
        mode_key = list(data.keys())[0]

    mode_data = data[mode_key]

    for hash_key, item in mode_data.items():
        question = item.get("original_prompt", "")
        answer = item.get("return", "")

        if dataset == 'pubmedqa':
            answer = parse_pubmedqa_answer(answer)

        if question and answer:
            qa_pairs.append({
                "question": question,
                "answer": answer
            })

    return qa_pairs


def load_qa_file(qa_file):
    """Load QA file and create question -> reference answers mapping."""
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    qa_map = {}
    for item in qa_data:
        question = item.get("question", "")
        if not question:
            continue

        # Get reference answers
        refs = []
        if 'ideal_answer' in item:
            refs = item['ideal_answer']
            if isinstance(refs, str):
                refs = [refs]
        elif 'answer_long' in item:
            refs = [item['answer_long']] if item['answer_long'] else []

        qa_map[question] = {
            "refs": refs if refs else [],
            "type": item.get("type", "summary")
        }

    return qa_map


def compute_rouge_l(references, hypothesis):
    """Compute ROUGE-L score between references and hypothesis."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    if not references or not hypothesis:
        return {"rougeL_precision": 0.0, "rougeL_recall": 0.0, "rougeL_fmeasure": 0.0}

    best_score = {"rougeL_precision": 0.0, "rougeL_recall": 0.0, "rougeL_fmeasure": 0.0}
    for ref in references:
        scores = scorer.score(ref, hypothesis)
        if scores['rougeL'].fmeasure > best_score["rougeL_fmeasure"]:
            best_score = {
                "rougeL_precision": scores['rougeL'].precision,
                "rougeL_recall": scores['rougeL'].recall,
                "rougeL_fmeasure": scores['rougeL'].fmeasure
            }
    return best_score


def compute_bertscore_batch(references_list, hypothesis_list, lang='en'):
    """Compute BERTScore for a batch of references and hypotheses."""
    if not BERTSCORE_AVAILABLE:
        print("BERTScore not available, skipping.")
        return None

    if not references_list or not hypothesis_list:
        return None

    all_refs = []
    all_hyps = []

    for refs, hyp in zip(references_list, hypothesis_list):
        if refs and hyp:
            all_refs.append(refs[0])
            all_hyps.append(hyp)

    if not all_refs:
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Computing BERTScore on {device} with {len(all_refs)} pairs...")

    try:
        P, R, F1 = bert_score(all_hyps, all_refs, lang=lang, verbose=False, device=device)
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item()
        }
    except Exception as e:
        print(f"  BERTScore computation failed: {e}")
        return None


def evaluate(answer_file, qa_file, dataset, skip_bertscore=False):
    """Evaluate using ROUGE-L and optionally BERTScore."""
    # Load data
    qa_pairs = load_answer_file(answer_file, dataset)
    qa_map = load_qa_file(qa_file)

    # Match answers with references
    references_list = []
    hypothesis_list = []
    questions = []
    query_types = []

    matched = 0
    unmatched = 0

    for qa_pair in qa_pairs:
        question = qa_pair["question"]
        answer = qa_pair["answer"]

        if question in qa_map:
            refs = qa_map[question]["refs"]
            qa_type = qa_map[question]["type"]

            if refs and answer:
                questions.append(question)
                query_types.append(qa_type)
                references_list.append(refs)
                hypothesis_list.append(answer)
                matched += 1
        else:
            unmatched += 1

    n = len(questions)
    print(f"Total answer pairs: {len(qa_pairs)}")
    print(f"Matched with QA file: {matched}")
    print(f"Unmatched: {unmatched}")
    print(f"Valid pairs with references: {n}")

    if n == 0:
        print("No valid pairs found. Exiting.")
        return None

    # Compute ROUGE-L
    print("Computing ROUGE-L...")
    rouge_results = []
    rouge_avgs = {"rougeL_precision": 0.0, "rougeL_recall": 0.0, "rougeL_fmeasure": 0.0}

    for refs, hyp in zip(references_list, hypothesis_list):
        scores = compute_rouge_l(refs, hyp)
        rouge_results.append(scores)
        for k in rouge_avgs:
            rouge_avgs[k] += scores[k]

    for k in rouge_avgs:
        rouge_avgs[k] /= n

    # Compute BERTScore (optional)
    bert_scores = None
    if not skip_bertscore:
        print("Computing BERTScore...")
        bert_scores = compute_bertscore_batch(references_list, hypothesis_list)
    else:
        print("Skipping BERTScore (--no_bertscore flag).")

    # Print overall results
    print("\n" + "=" * 60)
    print("Overall N-gram Metrics Results")
    print("=" * 60)
    print(f"  ROUGE-L Precision:   {rouge_avgs['rougeL_precision']:.4f}")
    print(f"  ROUGE-L Recall:      {rouge_avgs['rougeL_recall']:.4f}")
    print(f"  ROUGE-L F1:          {rouge_avgs['rougeL_fmeasure']:.4f}")
    if bert_scores:
        print(f"  BERTScore Precision: {bert_scores['bertscore_precision']:.4f}")
        print(f"  BERTScore Recall:    {bert_scores['bertscore_recall']:.4f}")
        print(f"  BERTScore F1:        {bert_scores['bertscore_f1']:.4f}")
    print(f"  Num Questions:       {n}")

    # Per-type breakdown
    type_stats = defaultdict(lambda: {"count": 0, "rougeL_sum": 0.0})
    for i, qt in enumerate(query_types):
        type_stats[qt]["count"] += 1
        type_stats[qt]["rougeL_sum"] += rouge_results[i]["rougeL_fmeasure"]

    if type_stats:
        print("\n" + "=" * 60)
        print("Per-Type Breakdown (ROUGE-L F1)")
        print("=" * 60)
        for qt, stats in sorted(type_stats.items()):
            count = stats["count"]
            avg_rouge = stats["rougeL_sum"] / count if count > 0 else 0
            print(f"  {qt:10s}: count={count:3d}, ROUGE-L F1={avg_rouge:.4f}")

    # Build output
    output = {
        "dataset": dataset,
        "overall": {
            "rougeL_precision": rouge_avgs["rougeL_precision"],
            "rougeL_recall": rouge_avgs["rougeL_recall"],
            "rougeL_fmeasure": rouge_avgs["rougeL_fmeasure"],
            "num_questions": n,
        },
        "per_question": []
    }

    if bert_scores:
        output["overall"]["bertscore_precision"] = bert_scores["bertscore_precision"]
        output["overall"]["bertscore_recall"] = bert_scores["bertscore_recall"]
        output["overall"]["bertscore_f1"] = bert_scores["bertscore_f1"]

    for i in range(n):
        item = {
            "question": questions[i],
            "type": query_types[i],
            "rougeL_fmeasure": rouge_results[i]["rougeL_fmeasure"],
        }
        output["per_question"].append(item)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Compute ROUGE-L and BERTScore for long-form text evaluation"
    )
    parser.add_argument('--dataset', type=str, required=True, choices=['bioasq', 'pubmedqa'],
                        help='Dataset type (bioasq or pubmedqa)')
    parser.add_argument('--qa_file', type=str, required=True,
                        help='Path to QA data file (e.g., ./data/bioasq_qa.json)')
    parser.add_argument('--answer_file', type=str, required=True,
                        help='Path to answer file (kv_store_llm_response_cache_all_ideal_*.json)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output JSON file for n-gram metrics')
    parser.add_argument('--no_bertscore', action='store_true',
                        help='Skip BERTScore computation (useful when model is not cached)')
    args = parser.parse_args()

    output = evaluate(args.answer_file, args.qa_file, args.dataset,
                      skip_bertscore=args.no_bertscore)

    if output:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()
