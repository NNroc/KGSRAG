#!/bin/bash

model_log=Qwen2.5-7B-Instruct
suffix=""
working_dir="./output/bioasq_${model_log}${suffix}"
golden_file="./evaluate/flat/BioASQEvaluation/evaluation/12B_golden.json"
system_response="./evaluate/flat/BioASQEvaluation/evaluation/system_response.json"
eval_result_dir="./output/log"
mkdir -p "$eval_result_dir"

run_eval() {
    local exp_name=$1
    local cache_file=$2

    echo "------------------------------------------"
    echo "Evaluating: $exp_name"
    echo "------------------------------------------"

    python evaluate/bioasq_evaluate.py \
        --answer_model "$model_log" \
        --answer_filename "$cache_file" \
        --mode all \
        --suffix "$suffix" \
        2>&1 | tee -a "${eval_result_dir}/eval_hp_${exp_name}.txt"

    java -Xmx10G -cp $CLASSPATH:./evaluate/flat/BioASQEvaluation/dist/BioASQEvaluation.jar \
        evaluation.EvaluatorTask1b -phaseB -e 5 \
        "$golden_file" "$system_response" -verbose \
        2>&1 | tee -a "${eval_result_dir}/eval_hp_${exp_name}.txt"

    echo ""
}

# --- Default baseline (T_min=0.1, T_max=0.5, alpha=10, beta=5) ---

# --- T_min experiments ---
run_eval "tmin02" "kv_store_llm_response_cache_hp_tmin02.json"
run_eval "tmin03" "kv_store_llm_response_cache_hp_tmin03.json"

# --- T_max experiments ---
run_eval "tmax04" "kv_store_llm_response_cache_hp_tmax04.json"
run_eval "tmax06" "kv_store_llm_response_cache_hp_tmax06.json"

# --- alpha experiments ---
run_eval "alpha3" "kv_store_llm_response_cache_hp_alpha3.json"
run_eval "alpha5" "kv_store_llm_response_cache_hp_alpha5.json"
run_eval "alpha15" "kv_store_llm_response_cache_hp_alpha15.json"
run_eval "alpha20" "kv_store_llm_response_cache_hp_alpha20.json"

# --- beta experiments ---
run_eval "beta3" "kv_store_llm_response_cache_hp_beta3.json"
run_eval "beta7" "kv_store_llm_response_cache_hp_beta7.json"
run_eval "beta10" "kv_store_llm_response_cache_hp_beta10.json"

echo "=========================================="
echo "All evaluations completed"
echo "Results saved to: ${eval_result_dir}/eval_hp_*.txt"
echo "=========================================="
