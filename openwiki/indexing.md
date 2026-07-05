# Indexing — `index.py`

Chunks `articles.jsonl`, embeds, and stores into the configured vector store.
This is the throughput-critical stage — see [benchmarking.md](benchmarking.md)
for the tuning story.

## The 3-stage parallel pipeline

```
producer ──batch_q──► embed worker ──store_q──► store thread ──► vector store
```

Threads communicate via bounded `queue.Queue`s. A slow stage blocks the
fast one (backpressure) instead of unbounded buffering.

### Producer thread (`run_producer`, `index.py:152`)

- Reads `articles.jsonl` line by line.
- Chunks with `RecursiveCharacterTextSplitter` (`CHUNK_SIZE=2000`,
  `CHUNK_OVERLAP=200`).
- Skips articles already fully stored (last chunk id exists in the vector
  store) — the cheapest resume path.
- Skips articles whose title hash is in `processed_titles` (resume cache).
- Accumulates `(id, title, url, text)` tuples into a batch of `batch_size`
  and pushes to `batch_q`. Sends a `None` sentinel when done.

### Embed worker (`embed_worker`, `index.py:58`)

- `PARALLEL_EMBED_WORKERS` threads (default 1 — MPS doesn't multi-process
  well; increasing it usually hurts more than it helps).
- Pulls a batch, calls `embedder.embed_documents(texts)` → `np.float32` 2D array.
- Pushes `(ids, metas, docs, vectors)` to `store_q`.
- On exception, appends to shared `errors` list and sends a sentinel so the
  store thread can drain and the main thread can re-raise.

### Store thread (`run_store`, `index.py:207`)

Behavior depends on `VECTOR_STORE`:

**FAISS (default)**
- Drains `store_q`, accumulates into `s_ids/s_metas/s_docs/s_vecs`.
- Every `flush_size` chunks: `np.vstack`s the vectors, L2-normalizes them,
  appends to a `faiss.IndexHNSWFlat`, inserts metadata into sqlite, and
  persists both `faiss_index.bin` and `metadata.sqlite`.
- FAISS bulk builds are much faster than ChromaDB's incremental HNSW inserts.

**ChromaDB**
- Drains `store_q`, accumulates into `s_ids/s_metas/s_docs/s_vecs`.
- Every `flush_size` chunks: `np.vstack`s and calls `collection.upsert(...)`,
  splitting into `MAX_UPSERT=5000` blocks because ChromaDB caps a single
  upsert around ~5461 elements.

## Resume and id scheme

Chunk ids are deterministic:

```python
id = f"{stable_hash(title)}::{chunk_index}"
# stable_hash = blake2b(title, digest_size=16).hexdigest()
```

This gives two resume mechanisms:

1. **Vector store ids** — for ChromaDB, every id already in the collection,
   fetched once at startup; for FAISS, every id in `metadata.sqlite`. The
   last chunk id of an article is checked first, before re-embedding.
2. **`processed_titles.json`** — set of title-hashes already fully stored,
   persisted every `SAVE_INTERVAL` flushes to reduce I/O.

`--reset` wipes the vector store directory (`vector_store/` or `chroma_db/`)
and `processed_titles.json` for a clean start.

## CLI

```bash
python3 index.py [--reset] [--encode-batch-size N] [--flush-size N]
                 [--hnsw-ef N] [--save-interval N] [--vector-store {chroma,faiss}]
```

| Flag | Default | Overrides |
|------|---------|-----------|
| `--reset` | off | Delete vector store + `processed_titles.json`, reindex from scratch |
| `--encode-batch-size` | `2048` (`EMBEDDING_BATCH_SIZE`) | Batch size for `SentenceTransformer.encode` |
| `--flush-size` | `5000` (`STORAGE_FLUSH_SIZE`) | Vectors per store flush |
| `--hnsw-ef` | `40` (`HNSW_EF_CONSTRUCTION`) | HNSW `ef_construction` (also used for `FAISS_HNSW_EF_CONSTRUCTION`) |
| `--save-interval` | `5` (`SAVE_INTERVAL`) | Persist `processed_titles.json` every N flushes |
| `--vector-store` | `faiss` (`VECTOR_STORE`) | Backend used for this run |

### Fastest known config (M1, 1000 articles, ~15.6k chunks)

```bash
python3 index.py --reset --encode-batch-size=2048 --flush-size=5000
```

~1160 chunks/s, ~13s for ~15.6k chunks with the default FAISS + L3 model.
The same dataset with the old ChromaDB + L6 setup takes ~54s (~288 chunks/s).

## Tuning knobs (and their ceilings)

- **`--encode-batch-size`**: bigger = better GPU util, more RAM. 2048 is a
  sweet spot on M1 16GB. Too big → MPS OOM.
- **`--flush-size`**: bigger = fewer sqlite/FAISS or upsert round-trips.
  ChromaDB is still capped internally by `MAX_UPSERT=5000`; FAISS has no
  hard cap but commits metadata per flush.
- **`--hnsw-ef`**: 40 vs ChromaDB default 100 → ~2× faster insert at the
  cost of some recall. For a bulk-load of a static dump, recall is rarely the
  bottleneck; for a query-heavy deployment you'd raise this.
- **`PARALLEL_EMBED_WORKERS`**: leave at 1. MPS serializes anyway; more
  threads just add contention. Only bump if you move to CUDA with multiple
  GPUs.
- **`QUEUE_MAXSIZE`**: 4. Small on purpose — backpressure keeps memory flat.
