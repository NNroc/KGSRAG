# shellcheck disable=SC2034
model_name=/data/pretrained/Qwen/Qwen2.5-7B-Instruct
model_log=Qwen2.5-7B-Instruct
python src/question_decomposition.py --dataset pubmedqa --decomposition statement --model_name "$model_name"
python src/question_decomposition.py --dataset bioasq --decomposition statement --model_name "$model_name"

model_name=/data/pretrained/Qwen/Qwen2.5-14B-Instruct
model_log=Qwen2.5-14B-Instruct
python src/question_decomposition.py --dataset pubmedqa --decomposition statement --model_name "$model_name"
python src/question_decomposition.py --dataset bioasq --decomposition statement --model_name "$model_name"

model_name=/data/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct
model_log=Meta-Llama-3.1-8B-Instruct
python src/question_decomposition.py --dataset pubmedqa --decomposition statement --model_name "$model_name"
python src/question_decomposition.py --dataset bioasq --decomposition statement --model_name "$model_name"
