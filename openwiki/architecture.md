# Architecture

## The pipeline end to end

```
enwiki dump (.xml.bz2)
        │
        ▼
   ingest.py ──────► articles.jsonl
   (XML parse +          {title, text, url}
    mwparserfromhell)
        │
        ▼
   index.py ──────► vector_store/
   (chunk + embed      faiss_index.bin
    + store)          metadata.sqlite
        │              processed_titles.json
        │                 (resume cache)
        ▼
   app.py  ◄── rag.py
   (Streamlit)  (retriever + RAG chain)
        │
        ▼
   LM Studio (localhost:1234)
   chat LLM
```

Three stages, three commands. Each stage is independent and resumable:
`articles.jsonl` is a plain JSONL file, `vector_store/` is a FAISS index plus
sqlite metadata, and `processed_titles.json` lets the indexer skip articles it
has already embedded. The legacy `chroma_db/` backend is still available via
`config.VECTOR_STORE = "chroma"`.

## Module map

| File | Role | Key symbols |
|------|------|-------------|
| `config.py` | All tunable parameters in one place | `EMBEDDING_MODEL_NAME`, `VECTOR_STORE`, `CHUNK_SIZE`, ... |
| `ingest.py` | Dump → articles.jsonl | `clean_wikitext()`, `main()` |
| `embeddings.py` | Two `Embeddings` impls | `LMStudioEmbeddings`, `LocalEmbeddings` |
| `index.py` | Parallel chunk+embed+store pipeline | `stable_hash()`, `build_embedder()`, `embed_worker()`, `main()` |
| `rag.py` | Build retriever + RAG chain | `build_retriever()`, `build_rag_chain()` |
| `app.py` | Streamlit UI | `get_retriever()` |
| `benchmark.py` | Time embedding batch sizes | `sample_chunks()`, `time_backend()` |

## The indexing pipeline (the interesting part)

`index.py` is a 3-stage thread cascade with bounded queues for backpressure:

```
producer thread          embed worker(s)          store thread
─────────────────        ─────────────────        ─────────────────
read JSONL line          pull batch from          drain store_q
chunk (splitter)         batch_q                  accumulate up to
skip already-processed   embed_documents()        flush_size chunks
push batch ─► batch_q    push (ids,metas,         FAISS / ChromaDB
                         docs,vectors) ─► store_q persist metadata
```

- Queues are bounded (`QUEUE_MAXSIZE`, default 4) so a slow stage doesn't
  let a fast one blow up memory.
- `store_q` is intentionally larger (2×) because the store flush is async.
- Chunk ids are deterministic (`stable_hash(title)::chunk_index`), so resume
  is just "skip ids already in the store". See
  [indexing.md](indexing.md#resume-and-id-scheme).

## Embedding: local vs LM Studio

Two implementations of LangChain's `Embeddings` interface live in
`embeddings.py`. `config.USE_LOCAL_EMBEDDING` selects which one both
`index.py` and `rag.py` use. **They must match** — query embeddings and
document embeddings must come from the same model/dimensions or retrieval
returns garbage. Local is the default and much faster on Apple Silicon. See
[embeddings.md](embeddings.md).

## Vector store: FAISS vs ChromaDB

The default vector store is now FAISS + sqlite metadata (`config.VECTOR_STORE
= "faiss"`). FAISS builds a one-shot HNSW index during bulk load, which is
significantly faster than ChromaDB's incremental HNSW inserts. ChromaDB is
still supported by setting `VECTOR_STORE = "chroma"` in `config.py` (and
passing `--vector-store chroma` to `index.py`).

## RAG chain

`rag.py` wires a LangChain `create_retrieval_chain` over a retriever
(either FAISS-backed or ChromaDB-backed, depending on `VECTOR_STORE`) and a
`ChatOpenAI` pointed at LM Studio. The prompt stuffs the retrieved context
and asks for a concise answer with source titles. `app.py` is a thin
Streamlit wrapper that caches the retriever and rebuilds the chain only when
the selected model changes. See [rag-and-ui.md](rag-and-ui.md).

## Design notes

- **No external vector DB service.** FAISS and sqlite run embedded in-process.
- **No separate embedding service.** The same `LocalEmbeddings` class is
  used at index time and query time, ensuring dimensional compatibility.
- **Resume-friendly by design.** Deterministic ids + a processed-titles cache
  mean a crashed run picks up where it left off without re-embedding.
- **Apple Silicon first.** MPS + fp16 + the smaller L3 model give the best
  throughput on a Mac.
