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

# Embedding backend: "torch" (default) or "onnx" (faster for small models on CPU).
# "onnx" requires: pip install optimum onnxruntime
EMBEDDING_BACKEND = "torch"

# Parallel indexing pipeline (producer/embed/store threads).
PARALLEL_EMBED_WORKERS = 1   # embed worker threads (keep 1, MPS doesn't multi-process well)
QUEUE_MAXSIZE = 4            # bounded queue size between stages (backpressure)
STORAGE_FLUSH_SIZE = 5000    # upsert to ChromaDB every N embedded chunks
