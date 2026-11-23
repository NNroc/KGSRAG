#!/bin/bash

model_name=/home/npy/models/Qwen2.5-7B-Instruct
suffix=""
dataset=bioasq

run_exp() {
    local name=$1; shift
    CUDA_VISIBLE_DEVICES=0 $PY src/generate_answer.py \
        --dataset "$dataset" --mode all --suffix "$suffix" --dynamic_threshold \
        --model_name "$model_name" --cache_suffix "hp_${name}" "$@"
}

run_exp tmax04  --min_threshold 0.1 --max_threshold 0.4 --alpha 10 --beta 5
run_exp tmax06  --min_threshold 0.1 --max_threshold 0.6 --alpha 10 --beta 5

run_exp tmin02  --min_threshold 0.2 --max_threshold 0.5 --alpha 10 --beta 5
run_exp tmin03  --min_threshold 0.3 --max_threshold 0.5 --alpha 10 --beta 5

run_exp alpha3  --min_threshold 0.1 --max_threshold 0.5 --alpha 3  --beta 5
run_exp alpha5  --min_threshold 0.1 --max_threshold 0.5 --alpha 5  --beta 5
run_exp alpha15 --min_threshold 0.1 --max_threshold 0.5 --alpha 15 --beta 5

run_exp beta3   --min_threshold 0.1 --max_threshold 0.5 --alpha 10 --beta 3
run_exp beta7   --min_threshold 0.1 --max_threshold 0.5 --alpha 10 --beta 7
run_exp beta10  --min_threshold 0.1 --max_threshold 0.5 --alpha 10 --beta 10
