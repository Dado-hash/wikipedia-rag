# Configuration — `config.py`

Every tunable lives in `config.py`. `index.py` CLI flags override several of
them per-run; see [indexing.md](indexing.md#cli).

## LM Studio / chat

| Param | Default | Notes |
|-------|---------|-------|
| `LM_STUDIO_URL` | `http://localhost:1234/v1` | Chat LLM endpoint. Embeddings too when `USE_LOCAL_EMBEDDING=False`. |
| `EMBEDDING_MODEL` | `text-embedding-all-minilm-l6-v2-embedding` | Model name for the LM Studio embeddings API (only used in non-local mode) |
| `AVAILABLE_CHAT_MODELS` | `[gemma-4-12b-qat, lfm2.5-1.2b, nemotron-3-nano-4b, gemma-4-e2b]` | Sidebar dropdown in `app.py` |
| `DEFAULT_CHAT_MODEL` | `AVAILABLE_CHAT_MODELS[0]` | Used when `build_rag_chain` gets no `model_name` |

## Embeddings

| Param | Default | Notes |
|-------|---------|-------|
| `USE_LOCAL_EMBEDDING` | `True` | `True` → `LocalEmbeddings` (sentence-transformers/MPS). `False` → `LMStudioEmbeddings`. **Must match between index and query time.** |
| `EMBEDDING_USE_FP16` | `True` | Half precision on MPS. Ignored on CPU. |
| `EMBEDDING_BATCH_SIZE` | `2048` | Batch for `SentenceTransformer.encode`. Overridable via `--encode-batch-size`. |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/paraphrase-MiniLM-L3-v2` | Local model. Faster than `all-MiniLM-L6-v2` and has a 512-token context, so `CHUNK_SIZE=2000` fits without truncation. Swapping models requires a full reindex (same dimensions are a *must*). |

## Chunking & retrieval

| Param | Default | Notes |
|-------|---------|-------|
| `CHUNK_SIZE` | `2000` | Chars per chunk (`RecursiveCharacterTextSplitter`) |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Documents retrieved per query (retriever `k`) |

## Vector store

| Param | Default | Notes |
|-------|---------|-------|
| `VECTOR_STORE` | `"faiss"` | `"faiss"` builds a one-shot FAISS index + sqlite metadata. `"chroma"` keeps the legacy ChromaDB behavior. Must match between `index.py` and `rag.py`. |
| `VECTOR_STORE_DIR` | `"vector_store"` | Directory for the FAISS index and metadata DB. |
| `FAISS_INDEX_FILE` | `vector_store/faiss_index.bin` | Persisted FAISS index. |
| `METADATA_DB_FILE` | `vector_store/metadata.sqlite` | sqlite table mapping FAISS position → chunk id/title/url/text. |
| `FAISS_USE_HNSW` | `True` | Use HNSW index. `False` uses a flat IP index (faster build, slower search at scale). |
| `FAISS_HNSW_M` | `64` | HNSW `M` parameter. |
| `FAISS_HNSW_EF_CONSTRUCTION` | `40` | Lower = faster bulk build, lower recall. |
| `FAISS_HNSW_EF_SEARCH` | `64` | Query-time HNSW `efSearch`. |
| `CHROMA_DB_DIR` | `"chroma_db"` | Persistent ChromaDB directory (used when `VECTOR_STORE="chroma"`). |
| `HNSW_EF_CONSTRUCTION` | `40` | Same as above, used only for ChromaDB. |

## Indexing pipeline

| Param | Default | Notes |
|-------|---------|-------|
| `ARTICLES_FILE` | `articles.jsonl` | Input to `index.py`, output of `ingest.py` |
| `PARALLEL_EMBED_WORKERS` | `1` | Embed worker threads. Keep 1 on MPS. |
| `QUEUE_MAXSIZE` | `4` | Bounded queue between stages (backpressure). `store_q` is 2× this. |
| `STORAGE_FLUSH_SIZE` | `5000` | Vectors per store flush. |
| `SAVE_INTERVAL` | `5` | Persist `processed_titles.json` every N flushes. |

## Changing config

`config.py` is plain Python with no env-var layer — edit the file directly.
There is no `.env` support; if you need per-environment config, the laziest
path is to branch on `os.environ` at the top of `config.py` rather than
introducing a config library.
