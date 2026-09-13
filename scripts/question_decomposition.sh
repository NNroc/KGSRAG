# shellcheck disable=SC2034
model_name=/home/npy/models/Qwen2.5-7B-Instruct
model_log=Qwen2.5-7B-Instruct
python src/question_decomposition.py --dataset pubmedqa --decomposition statement --model_name "$model_name"
python src/question_decomposition.py --dataset bioasq --decomposition statement --model_name "$model_name"

model_name=/home/npy/models/Qwen2.5-14B-Instruct
model_log=Qwen2.5-14B-Instruct
python src/question_decomposition.py --dataset pubmedqa --decomposition statement --model_name "$model_name"
python src/question_decomposition.py --dataset bioasq --decomposition statement --model_name "$model_name"

model_name=/home/npy/models/Meta-Llama-3.1-8B-Instruct
model_log=Meta-Llama-3.1-8B-Instruct
python src/question_decomposition.py --dataset pubmedqa --decomposition statement --model_name "$model_name"
python src/question_decomposition.py --dataset bioasq --decomposition statement --model_name "$model_name"
