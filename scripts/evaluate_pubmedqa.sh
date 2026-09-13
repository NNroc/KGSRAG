model_name=/home/npy/models/Qwen2.5-7B-Instruct
python src/generate_answer.py --dataset pubmedqa --save_dir "./output/kgsrag_output" --mode all --suffix "" --dynamic_threshold --model_name "$model_name" --eval_file "kv_store_llm_response_cache_all.json"
model_name=/home/npy/models/Qwen2.5-14B-Instruct
python src/generate_answer.py --dataset pubmedqa --save_dir "./output/kgsrag_output" --mode all --suffix "" --dynamic_threshold --model_name "$model_name" --eval_file "kv_store_llm_response_cache_all.json"
model_name=/home/npy/models/Meta-Llama-3.1-8B-Instruct
python src/generate_answer.py --dataset pubmedqa --save_dir "./output/kgsrag_output" --mode all --suffix "" --dynamic_threshold --model_name "$model_name" --eval_file "kv_store_llm_response_cache_all.json"

# evaluate ideal answer through llm
python evaluate/long_evaluate.py --qa_file './data/pubmedqa_qa.json' --ga_file './output/kgsrag_output/pubmedqa_Qwen2.5-7B-Instruct/kv_store_llm_response_cache_all.json' --output_file './output/kgsrag_output/pubmedqa_Qwen2.5-7B-Instruct-kv_store_llm_response_cache_all.json'
python evaluate/long_evaluate.py --qa_file './data/pubmedqa_qa.json' --ga_file './output/kgsrag_output/pubmedqa_Qwen2.5-14B-Instruct/kv_store_llm_response_cache_all.json' --output_file './output/kgsrag_output/pubmedqa_Qwen2.5-14B-Instruct-kv_store_llm_response_cache_all.json'
python evaluate/long_evaluate.py --qa_file './data/pubmedqa_qa.json' --ga_file './output/kgsrag_output/pubmedqa_Meta-Llama-3.1-8B-Instruct/kv_store_llm_response_cache_all.json' --output_file './output/kgsrag_output/pubmedqa_Meta-Llama-3.1-8B-Instruct-kv_store_llm_response_cache_all.json'

# evaluate ideal answer through ROUGE-L and BERTScore
python evaluate/ngram_eval.py \
    --dataset pubmedqa \
    --qa_file './data/pubmedqa_qa.json' \
    --answer_file './output/kgsrag_output/pubmedqa_Qwen2.5-7B-Instruct/kv_store_llm_response_cache_all.json' \
    --output_file './output/kgsrag_output/pubmedqa_Qwen2.5-7B-Instruct_ngram_metrics_all.json'
python evaluate/ngram_eval.py \
    --dataset pubmedqa \
    --qa_file './data/pubmedqa_qa.json' \
    --answer_file './output/kgsrag_output/pubmedqa_Qwen2.5-14B-Instruct/kv_store_llm_response_cache_all.json' \
    --output_file './output/kgsrag_output/pubmedqa_Qwen2.5-14B-Instruct_ngram_metrics_all.json'
python evaluate/ngram_eval.py \
    --dataset pubmedqa \
    --qa_file './data/pubmedqa_qa.json' \
    --answer_file './output/kgsrag_output/pubmedqa_Meta-Llama-3.1-8B-Instruct/kv_store_llm_response_cache_all.json' \
    --output_file './output/kgsrag_output/pubmedqa_Meta-Llama-3.1-8B-Instruct_ngram_metrics_all.json'
