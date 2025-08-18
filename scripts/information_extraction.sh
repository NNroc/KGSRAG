# create knowledge graph
# shellcheck disable=SC2034

suffix=-2
model_log=Qwen2.5-7B-Instruct-2
model_name=/data/pretrained/Qwen/Qwen2.5-7B-Instruct
python src/information_extraction.py --dataset pubmedqa --model_name "$model_name" --suffix "$suffix"
python src/information_extraction.py --dataset bioasq --model_name "$model_name" --suffix "$suffix"

model_log=Qwen2.5-14B-Instruct-2
model_name=/data/pretrained/Qwen/Qwen2.5-14B-Instruct
python src/information_extraction.py --dataset pubmedqa --model_name "$model_name" --suffix "$suffix"
python src/information_extraction.py --dataset bioasq --model_name "$model_name" --suffix "$suffix"

model_log=Meta-Llama-3.1-8B-Instruct-2
model_name=/data/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct
python src/information_extraction.py --dataset pubmedqa --model_name "$model_name" --suffix "$suffix"
python src/information_extraction.py --dataset bioasq --model_name "$model_name" --suffix "$suffix"
