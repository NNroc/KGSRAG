# KGSRAG
Code for KGSRAG: Retrieval-Augmented Generation System for Biomedical Information Retrieval and Reasoning Based on Knowledge Graphs and Statements.[TechRxiv](https://www.techrxiv.org/doi/full/10.36227/techrxiv.175735355.54231550/v1)

## Dataset
The [BioASQ-QA](https://pmc.ncbi.nlm.nih.gov/articles/PMC10042099/pdf/41597_2023_Article_2068.pdf) dataset can be downloaded following the instructions at [here](https://participants-area.bioasq.org/datasets/). 
The [PubMedQA](https://aclanthology.org/D19-1259.pdf) dataset can be downloaded following the instructions at [here](https://pubmedqa.github.io/). 
Several samples have been added to the data folder (`./data/BioASQ-QA/` and `./data/PubMedQA/`)

The expected structure of files is:
```
KGSRAG
 |-- data
 |    |-- BioASQ-QA
 |    |    |-- 12B1_golden.json
 |    |    |-- 12B2_golden.json
 |    |    |-- 12B3_golden.json
 |    |    |-- 12B4_golden.json
 |    |-- PubMedQA
 |    |    |-- ori_pqaa.json
 |    |    |-- ori_pqal.json
 |    |    |-- ori_pqau.json
 |    |-- bioasq_corpus.json
 |    |-- bioasq_qa.json
 |    |-- pubmedqa_corpus.json
 |    |-- pubmed_qa.json
```

## Environment
```
conda create -n KGSRAG python=3.9.20
conda activate KGSRAG
pip install -r requirements.txt
```

## Preprocessing Data
```
python ./preprocess/pre_bioasq.py
python ./preprocess/pre_pubmedqa.py
```

## Information Extraction
```
bash ./script/information_extraction.sh
```

## Question Extraction
```
bash ./script/question_decomposition.sh
```

## Answer Generation
```
bash script/generate_answer.sh
```

## Evaluation
```
bash script/evaluate_bioasq.sh
bash script/evaluate_pubmedqa.sh
```
