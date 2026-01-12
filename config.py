import torch

class Config:
    LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    EMBEDDING_MODEL_ID = "BAAI/bge-m3"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOAD_IN_4BIT = True
    SEED = 42
    MAX_NEW_TOKENS = 150
    TEST_SAMPLE_LIMIT = 12
    RAG_TOP_K = 3
    SURPRISE_THRESHOLD = 1.2