#python evaluate/bioasq_evaluate.py --mode all --filepath /home/npy/projects/KGSRAG/output/kgsrag_output/bioasq_Qwen2.5-7B-Instruct/kv_store_llm_response_cache_all.json
#java -Xmx10G -cp $CLASSPATH:./evaluate/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 ./evaluate/flat/BioASQEvaluation/evaluation/12B_golden.json ./evaluate/flat/BioASQEvaluation/evaluation/system_response.json -verbose
#python evaluate/bioasq_evaluate.py --mode all --filepath /home/npy/projects/KGSRAG/output/kgsrag_output/bioasq_Qwen2.5-14B-Instruct/kv_store_llm_response_cache_all.json
#java -Xmx10G -cp $CLASSPATH:./evaluate/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 ./evaluate/flat/BioASQEvaluation/evaluation/12B_golden.json ./evaluate/flat/BioASQEvaluation/evaluation/system_response.json -verbose
#python evaluate/bioasq_evaluate.py --mode all --filepath /home/npy/projects/KGSRAG/output/kgsrag_output/bioasq_Meta-Llama-3.1-8B-Instruct/kv_store_llm_response_cache_all.json
#java -Xmx10G -cp $CLASSPATH:./evaluate/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 ./evaluate/flat/BioASQEvaluation/evaluation/12B_golden.json ./evaluate/flat/BioASQEvaluation/evaluation/system_response.json -verbose
#
## evaluate ideal answer through llm
#python evaluate/long_evaluate.py --qa_file './data/bioasq_qa.json' --ga_file './output/kgsrag_output/bioasq_Qwen2.5-7B-Instruct/kv_store_llm_response_cache_all_ideal.json' --output_file './output/kgsrag_output/bioasq_Qwen2.5-7B-Instruct-kv_store_llm_response_cache_all_ideal.json'
#python evaluate/long_evaluate.py --qa_file './data/bioasq_qa.json' --ga_file './output/kgsrag_output/bioasq_Qwen2.5-14B-Instruct/kv_store_llm_response_cache_all_ideal.json' --output_file './output/kgsrag_output/bioasq_Qwen2.5-14B-Instruct-kv_store_llm_response_cache_all_ideal.json'
#python evaluate/long_evaluate.py --qa_file './data/bioasq_qa.json' --ga_file './output/kgsrag_output/bioasq_Meta-Llama-3.1-8B-Instruct/kv_store_llm_response_cache_all_ideal.json' --output_file './output/kgsrag_output/bioasq_Meta-Llama-3.1-8B-Instruct-kv_store_llm_response_cache_all_ideal.json'

# evaluate ideal answer through ROUGE-L and BERTScore
python evaluate/ngram_eval.py \
    --dataset bioasq \
    --qa_file './data/bioasq_qa.json' \
    --answer_file './output/kgsrag_output/bioasq_Qwen2.5-7B-Instruct/kv_store_llm_response_cache_all_ideal.json' \
    --output_file './output/kgsrag_output/bioasq_Qwen2.5-7B-Instruct_ngram_metrics_all_ideal.json'
python evaluate/ngram_eval.py \
    --dataset bioasq \
    --qa_file './data/bioasq_qa.json' \
    --answer_file './output/kgsrag_output/bioasq_Qwen2.5-14B-Instruct/kv_store_llm_response_cache_all_ideal.json' \
    --output_file './output/kgsrag_output/bioasq_Qwen2.5-14B-Instruct_ngram_metrics_all_ideal.json'
python evaluate/ngram_eval.py \
    --dataset bioasq \
    --qa_file './data/bioasq_qa.json' \
    --answer_file './output/kgsrag_output/bioasq_Meta-Llama-3.1-8B-Instruct/kv_store_llm_response_cache_all_ideal.json' \
    --output_file './output/kgsrag_output/bioasq_Meta-Llama-3.1-8B-Instruct_ngram_metrics_all_ideal.json'
