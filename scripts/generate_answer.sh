# shellcheck disable=SC2034
suffix=-2

##### PubMedQA #####
model_name=/data/pretrained/Qwen/Qwen2.5-7B-Instruct
python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name"

model_name=/data/pretrained/Qwen/Qwen2.5-14B-Instruct
python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name"

model_name=/data/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct
python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name"

##### BioASQ #####
model_name=/data/pretrained/Qwen/Qwen2.5-7B-Instruct
python src/generate_answer.py --dataset bioasq --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name"
python src/generate_answer.py --dataset bioasq --mode all --ideal --suffix "$suffix" --dynamic_threshold --model_name "$model_name"

model_name=/data/pretrained/Qwen/Qwen2.5-14B-Instruct
python src/generate_answer.py --dataset bioasq --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name"
python src/generate_answer.py --dataset bioasq --mode all --ideal --suffix "$suffix" --dynamic_threshold --model_name "$model_name"

model_name=/data/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct
python src/generate_answer.py --dataset bioasq --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name"
python src/generate_answer.py --dataset bioasq --mode all --ideal --suffix "$suffix" --dynamic_threshold --model_name "$model_name"
