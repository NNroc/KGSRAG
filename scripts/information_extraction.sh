# create knowledge graph
# shellcheck disable=SC2034

suffix=""

model_log=Qwen2.5-7B-Instruct
model_name=/data/pretrained/Qwen/Qwen2.5-7B-Instruct
python src/information_extraction.py --dataset pubmedqa --model_name "$model_name" --suffix "$suffix" 2>&1 | tee -a "./output/log/log_ie_pubmedqa_$model_log.txt"
python src/information_extraction.py --dataset bioasq --model_name "$model_name" --suffix "$suffix" 2>&1 | tee -a "./output/log/log_ie_bioasq_$model_log.txt"

model_log=Qwen2.5-14B-Instruct
model_name=/data/pretrained/Qwen/Qwen2.5-14B-Instruct
python src/information_extraction.py --dataset pubmedqa --model_name "$model_name" --suffix "$suffix" 2>&1 | tee -a "./output/log/log_ie_pubmedqa_$model_log.txt"
python src/information_extraction.py --dataset bioasq --model_name "$model_name" --suffix "$suffix" 2>&1 | tee -a "./output/log/log_ie_bioasq_$model_log.txt"

model_log=Meta-Llama-3.1-8B-Instruct
model_name=/data/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct
python src/information_extraction.py --dataset pubmedqa --model_name "$model_name" --suffix "$suffix" 2>&1 | tee -a "./output/log/log_ie_pubmedqa_$model_log.txt"
python src/information_extraction.py --dataset bioasq --model_name "$model_name" --suffix "$suffix" 2>&1 | tee -a "./output/log/log_ie_bioasq_$model_log.txt"
