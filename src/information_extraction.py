import os

os.environ["TRANSFORMERS_CACHE"] = "/data/pretrained/cache"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import json
import torch
import numpy as np
import random
import sys

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModel, AutoTokenizer
from rag import KGSRAG
from rag.llm import hf_model_complete, hf_embedding
from rag.utils import EmbeddingFunc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmedqa', help='medqa/pubmedqa/bioasq')
    parser.add_argument('--model_name', type=str, default='/data/pretrained/Qwen/Qwen2.5-7B-Instruct',
                        help='Specific model name or save path')
    parser.add_argument('--save_dir', type=str, default='./output', help='save path')
    parser.add_argument('--suffix', type=str, default='-2', help='-0/-1/-2')
    args = parser.parse_args()

    chunk_token_size = 1600
    chunk_overlap_token_size = 100
    dataset = args.dataset
    corpus = json.load(open(f'data/{dataset}_corpus.json', 'r'))
    model_name = args.model_name.split('/')[-1]
    WORKING_DIR = args.save_dir + '/' + dataset + "_" + model_name + args.suffix
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    rag = KGSRAG(
        dataset=dataset,
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name=args.model_name,
        chunk_token_size=chunk_token_size,
        chunk_overlap_token_size=chunk_overlap_token_size,
        entity_extract_max_gleaning=0,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=lambda texts: hf_embedding(
                texts,
                tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
                embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            ),
        ),
    )
    rag.insert(corpus)
    rag.generate_db()
