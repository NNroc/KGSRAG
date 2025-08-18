suffix=-2

model_name=/data/pretrained/Qwen/Qwen2.5-7B-Instruct
model_log=Qwen2.5-7B-Instruct
origin_path="./output/pubmedqa_Qwen2.5-7B-Instruct-2/kv_store_llm_response_cache.json"
python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name" --eval_file "kv_store_llm_response_cache_all.json"

model_name=/data/pretrained/Qwen/Qwen2.5-14B-Instruct
model_log=Qwen2.5-14B-Instruct
origin_path="./output/pubmedqa_Qwen2.5-14B-Instruct-2/kv_store_llm_response_cache.json"
python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name" --eval_file "kv_store_llm_response_cache_all.json"

model_name=/data/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct
model_log=Meta-Llama-3.1-8B-Instruct
origin_path="./output/pubmedqa_Meta-Llama-3.1-8B-Instruct-2/kv_store_llm_response_cache.json"
python src/generate_answer.py --dataset pubmedqa --mode all --suffix "$suffix" --dynamic_threshold --model_name "$model_name" --eval_file "kv_store_llm_response_cache_all.json"

# evaluate ideal answer
python evaluate/long_evaluate.py --qa_file './data/pubmedqa_qa.json' --ga_file './output/pubmedqa_Qwen2.5-7B-Instruct-2/kv_store_llm_response_cache_all.json'
python evaluate/long_evaluate.py --qa_file './data/pubmedqa_qa.json' --ga_file './output/pubmedqa_Qwen2.5-14B-Instruct-2/kv_store_llm_response_cache_all.json'
python evaluate/long_evaluate.py --qa_file './data/pubmedqa_qa.json' --ga_file './output/pubmedqa_Meta-Llama-3.1-8B-Instruct-2/kv_store_llm_response_cache_all.json'
