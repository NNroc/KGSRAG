# KGSRAG
Code for KGSRAG: Retrieval-Augmented Generation System for Biomedical Information Retrieval and Reasoning Based on Knowledge Graphs and Statements. [Information Sciences](https://doi.org/10.1016/j.ins.2026.124450), vol. 767, article 124450, 2027.

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
 |    |-- pubmedqa_qa.json
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
bash ./scripts/information_extraction.sh
```

## Question Extraction
```
bash ./scripts/question_decomposition.sh
```

## Answer Generation
```
bash scripts/generate_answer.sh
```

## Evaluation
```
bash scripts/evaluate_bioasq.sh
bash scripts/evaluate_pubmedqa.sh
```

## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{nie2027kgsrag,
  title   = {KGSRAG: Retrieval-augmented generation system for medical information retrieval and reasoning based on knowledge graphs and statements},
  author  = {Nie, Pengyuan and Lin, Mengxuan and Ma, Wenbing and Cong, Linlin and Yin, Rongling and Zu, Mian and Wu, Qingqiang and Wang, Lei},
  journal = {Information Sciences},
  volume  = {767},
  pages   = {124450},
  year    = {2027},
  doi     = {10.1016/j.ins.2026.124450}
}
```
