LM_STUDIO_URL = "http://localhost:1234/v1"
EMBEDDING_MODEL = "text-embedding-all-minilm-l6-v2-embedding"

AVAILABLE_CHAT_MODELS = [
    "google/gemma-4-12b-qat",
    "liquid/lfm2.5-1.2b",
    "nvidia/nemotron-3-nano-4b",
    "google/gemma-4-e2b",
]
DEFAULT_CHAT_MODEL = AVAILABLE_CHAT_MODELS[0]

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
TOP_K = 5

CHROMA_DB_DIR = "chroma_db"
ARTICLES_FILE = "articles.jsonl"

# Use local sentence-transformers (MPS/GPU) instead of LM Studio API
USE_LOCAL_EMBEDDING = True
EMBEDDING_USE_FP16 = True
EMBEDDING_BATCH_SIZE = 2048

# Local embedding model.  paraphrase-MiniLM-L3-v2 is ~3× faster than
# all-MiniLM-L6-v2 on Apple Silicon and has a 512-token context, so the
# default CHUNK_SIZE=2000 fits without truncation (L6 truncates to 256).
# Dimensions must still match between index and query time.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Vector store backend.  "faiss" builds a one-shot FAISS HNSW index backed by
# sqlite for metadata (much faster bulk builds than ChromaDB).  "chroma"
# keeps the previous ChromaDB behavior.
VECTOR_STORE = "faiss"
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_FILE = f"{VECTOR_STORE_DIR}/faiss_index.bin"
METADATA_DB_FILE = f"{VECTOR_STORE_DIR}/metadata.sqlite"
FAISS_HNSW_M = 64
FAISS_HNSW_EF_CONSTRUCTION = 40

# HNSW index tuning — lower ef_construction = faster bulk inserts.
# Default ChromaDB is 100; 40 is ~2× faster insert during bulk load.
HNSW_EF_CONSTRUCTION = 40

# Parallel indexing pipeline (producer/embed/store threads).
PARALLEL_EMBED_WORKERS = 1   # embed worker threads (keep 1, MPS doesn't multi-process well)
QUEUE_MAXSIZE = 4            # bounded queue size between stages (backpressure)
STORAGE_FLUSH_SIZE = 5000    # vectors per store flush

# Persist processed_titles cache to disk every N flushes (reduce lock / I/O pressure).
SAVE_INTERVAL = 5

# ── Advanced RAG techniques (Fase 1) ────────────────────────────────────────
# Toggle each technique independently.  They compose as:
#   HyDE( query → LLM → hypothetical doc → embedding )  [if ENABLE_HYDE]
#   → HybridSearch( FAISS + BM25 + RRF )                 [if ENABLE_HYBRID_SEARCH]
#   → CrossEncoderReRank                                 [if ENABLE_RERANKING]
ENABLE_HYDE = False           # Hypothetical Document Embedding
ENABLE_HYBRID_SEARCH = True   # Dense (FAISS) + Sparse (BM25) + RRF fusion
ENABLE_RERANKING = True       # Cross-encoder re-ranking on retrieved docs

HYBRID_TOP_K = 15             # Retrieve N per method before fusion
RRF_K = 60                    # RRF constant (higher = more weight to top ranks)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_TOP_K = 5      # Number of docs to keep after re-ranking

# ── Chunking & Embedding (Fase 2) ───────────────────────────────────────────
# Contextual Chunk Headers are always active: the article title is prepended to
# every chunk before embedding, so chunks about "the red planet" without
# mentioning "Mars" are still retrieved for Mars queries.
# Context Enrichment adds adjacent chunks around each retrieved chunk.
ENABLE_CONTEXT_ENRICHMENT = False
CONTEXT_WINDOW_SIZE = 1      # Number of chunks before/after each retrieved chunk
