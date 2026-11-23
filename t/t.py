import os

os.environ["TRANSFORMERS_CACHE"] = "/data/pretrained/cache"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import numpy as np
import random

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

from rag import KGSRAG, QueryParam
from rag.llm import hf_model_complete, hf_embedding, llama_cpp_model_complete
from rag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = KGSRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,
    llm_model_name="NousResearch/Meta-Llama-3.1-8B-Instruct",
    # llm_model_func=llama_cpp_model_complete,
    # llm_model_name="/data/pretrained/gguf/qwen2.5-7b-instruct.gguf",
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

# with open("./dickens/book.txt", "r", encoding="utf-8") as f:
#     rag.insert(f.read())

# Perform naive search
# print(rag.query("What are the top themes in this story?", param=QueryParam(mode="naive")))

# Perform local search
# print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))

# Perform global search
# print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

# Perform hybrid search
print(rag.query("What was the social and economic context of Harriet Tubman's life in the mid-19th century?",
                param=QueryParam(mode="hybrid")))
