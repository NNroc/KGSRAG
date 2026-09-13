# shellcheck disable=SC2034
suffix=""

##### PubMedQA #####
model_name=/home/npy/models/Qwen2.5-7B-Instruct
python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name"
model_name=/home/npy/models/Qwen2.5-14B-Instruct
python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name"
model_name=/home/npy/models/Meta-Llama-3.1-8B-Instruct
python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name"


##### BioASQ #####
model_name=/home/npy/models/Qwen2.5-7B-Instruct
origin_path="./output/bioasq_Qwen2.5-7B-Instruct/kv_store_llm_response_cache.json"
python src/generate_answer.py --dataset bioasq --mode all --ideal --suffix "$suffix" --dynamic_threshold --model_name "$model_name"
mv "$origin_path" "./output/bioasq_Qwen2.5-7B-Instruct/kv_store_llm_response_cache_all_ideal.json"
python src/generate_answer.py --dataset bioasq --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name"

model_name=/home/npy/models/Qwen2.5-14B-Instruct
origin_path="./output/bioasq_Qwen2.5-14B-Instruct/kv_store_llm_response_cache.json"
python src/generate_answer.py --dataset bioasq --mode all --suffix "$suffix" --dynamic_threshold --ideal --model_name "$model_name"
mv "$origin_path" "./output/bioasq_Qwen2.5-14B-Instruct/kv_store_llm_response_cache_all_ideal.json"
python src/generate_answer.py --dataset bioasq --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name"

model_name=/home/npy/models/Meta-Llama-3.1-8B-Instruct
origin_path="./output/bioasq_Meta-Llama-3.1-8B-Instruct/kv_store_llm_response_cache.json"
python src/generate_answer.py --dataset bioasq --mode all --ideal --suffix "$suffix" --dynamic_threshold --model_name "$model_name"
mv "$origin_path" "./output/bioasq_Meta-Llama-3.1-8B-Instruct/kv_store_llm_response_cache_all_ideal.json"
python src/generate_answer.py --dataset bioasq --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name"
