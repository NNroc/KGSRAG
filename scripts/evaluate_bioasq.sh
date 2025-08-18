suffix=-2
# evaluate bioasq
python evaluate/bioasq_evaluate.py --answer_model Qwen2.5-7B-Instruct --answer_filename kv_store_llm_response_cache_all.json --mode all --suffix "$suffix"
java -Xmx10G -cp $CLASSPATH:./evaluate/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 ./evaluate/flat/BioASQEvaluation/evaluation/12B_golden.json ./evaluate/flat/BioASQEvaluation/evaluation/system_response.json -verbose
python evaluate/bioasq_evaluate.py --answer_model Qwen2.5-14B-Instruct --answer_filename kv_store_llm_response_cache_all.json --mode all --suffix "$suffix"
java -Xmx10G -cp $CLASSPATH:./evaluate/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 ./evaluate/flat/BioASQEvaluation/evaluation/12B_golden.json ./evaluate/flat/BioASQEvaluation/evaluation/system_response.json -verbose
python evaluate/bioasq_evaluate.py --answer_model Meta-Llama-3.1-8B-Instruct --answer_filename kv_store_llm_response_cache_all.json --mode naive --suffix "$suffix"
java -Xmx10G -cp $CLASSPATH:./evaluate/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 ./evaluate/flat/BioASQEvaluation/evaluation/12B_golden.json ./evaluate/flat/BioASQEvaluation/evaluation/system_response.json -verbose

# evaluate ideal answer
python evaluate/long_evaluate.py --qa_file './data/bioasq_qa.json' --ga_file './output/bioasq_Qwen2.5-14B-Instruct-2/kv_store_llm_response_cache_all_ideal_6.json'
python evaluate/long_evaluate.py --qa_file './data/bioasq_qa.json' --ga_file './output/bioasq_Qwen2.5-7B-Instruct-2/kv_store_llm_response_cache_all_ideal_6.json'
python evaluate/long_evaluate.py --qa_file './data/bioasq_qa.json' --ga_file './output/bioasq_Meta-Llama-3.1-8B-Instruct-2/kv_store_llm_response_cache_all_ideal_8.json'
