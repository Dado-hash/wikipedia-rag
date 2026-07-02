LM_STUDIO_URL = "http://localhost:1234/v1"
EMBEDDING_MODEL = "text-embedding-all-minilm-l6-v2-embedding"
CHAT_MODEL = "google/gemma-4-12b-qat"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
TOP_K = 5

CHROMA_DB_DIR = "chroma_db"
ARTICLES_FILE = "articles.jsonl"

# Use local sentence-transformers (MPS/GPU) instead of LM Studio API
USE_LOCAL_EMBEDDING = True
EMBEDDING_USE_FP16 = True
EMBEDDING_BATCH_SIZE = 512
