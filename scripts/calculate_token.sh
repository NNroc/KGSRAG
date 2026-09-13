# shellcheck disable=SC2034
suffix=""

model_name=/home/npy/models/Qwen2.5-7B-Instruct
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode keyword --suffix "$suffix" --dynamic_threshold --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode naive --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode keyword --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#
model_name=/home/npy/models/Qwen2.5-14B-Instruct
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode keyword --suffix "$suffix" --dynamic_threshold --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode naive --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode keyword --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#
model_name=/data/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode keyword --suffix "$suffix" --dynamic_threshold --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode naive --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset pubmedqa --mode keyword --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"

# ideal
model_name=/home/npy/models/Qwen2.5-7B-Instruct
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode all --ideal --suffix "$suffix" --dynamic_threshold --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode keyword --ideal --suffix "$suffix" --dynamic_threshold --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode naive --ideal --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode all --ideal --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode keyword --ideal --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"

model_name=/home/npy/models/Qwen2.5-14B-Instruct
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode all --ideal --suffix "$suffix" --dynamic_threshold --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode keyword --ideal --suffix "$suffix" --dynamic_threshold --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode naive --ideal --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode all --ideal --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode keyword --ideal --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"

model_name=/data/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode all --ideal --suffix "$suffix" --dynamic_threshold --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode keyword --ideal --suffix "$suffix" --dynamic_threshold --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode naive --ideal --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode all --ideal --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
#CUDA_VISIBLE_DEVICES=0 python src/generate_answer.py --dataset bioasq --mode keyword --ideal --suffix "$suffix" --model_name "$model_name" 2>&1 | tee -a "./output/log/log_cal_tokens.txt"
