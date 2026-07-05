# Wikipedia RAG

Retrieval-Augmented Generation over the English Wikipedia dump, using LangChain + Streamlit + LM Studio.

## Prerequisites

- Python 3.9+
- [LM Studio](https://lmstudio.ai/) running on `localhost:1234`
  - **Chat model**: `google/gemma-4-12b-qat` (or any OpenAI-compatible model)
- sentence-transformers + faiss-cpu (installed via `requirements.txt`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Extract articles from the dump

```bash
python3 ingest.py --max-articles 5000
```

Estrae articoli da `enwiki-latest-pages-articles.xml.bz2`, pulisce il wikitext, salva in `articles.jsonl`.

### 2. Chunk, embed, and index

```bash
python3 index.py --reset
```

- **Chunking**: `RecursiveCharacterTextSplitter` (2000 caratteri, overlap 200)
- **Pipeline**: 3 thread in cascata — producer → embed worker → store thread (dedicato, store flush non blocca la pipeline)
- **Embedding**: sentence-transformers (`paraphrase-MiniLM-L3-v2`) via MPS (GPU Apple Silicon) — ~1100-1200 chunk/s
- **Storage**: FAISS HNSW index + sqlite metadata (`vector_store/`); opzionale `VECTOR_STORE="chroma"` per retrocompatibilità.

CLI disponibili:
- `--reset`: ricrea l'indice da capo
- `--encode-batch-size N`: sovrascrive `EMBEDDING_BATCH_SIZE` (default 2048)
- `--flush-size N`: sovrascrive `STORAGE_FLUSH_SIZE` (default 5000)
- `--hnsw-ef N`: HNSW `ef_construction` (default 40, più basso = insert più veloci)
- `--save-interval N`: salva `processed_titles` ogni N flush (default 5)
- `--vector-store {chroma,faiss}`: backend per questa run

> **Tuning**: la configurazione di default è già la più veloce testata:
> ```
> python3 index.py --reset --encode-batch-size=2048 --flush-size=5000
> ```
> Il modello L3 ha un contesto di 512 token, quindi chunk da 2000 caratteri non vengono più troncati.

### 3. Launch the UI

```bash
streamlit run app.py
```

Apri http://localhost:8501. Le query embedding usano lo stesso modello locale (compatibile con l'indice).

## Configurazione

Tutti i parametri in `config.py`:

| Parametro | Default | Note |
|-----------|---------|------|
| `LM_STUDIO_URL` | `http://localhost:1234/v1` | Endpoint chat LLM |
| `DEFAULT_CHAT_MODEL` | `google/gemma-4-12b-qat` | Modello per risposte |
| `USE_LOCAL_EMBEDDING` | `True` | Usa sentence-transformers (MPS) invece di LM Studio API |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/paraphrase-MiniLM-L3-v2` | Modello embedding (384 dim, 512 token context) |
| `EMBEDDING_BATCH_SIZE` | `2048` | Batch per `SentenceTransformer.encode` |
| `EMBEDDING_USE_FP16` | `True` | Mezza precisione su MPS |
| `CHUNK_SIZE` | `2000` | Caratteri per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap tra chunk consecutivi |
| `TOP_K` | `5` | Documenti recuperati per query |
| `VECTOR_STORE` | `"faiss"` | `"faiss"` o `"chroma"` |
| `FAISS_USE_HNSW` | `True` | `False` per indice flat (build più veloce, search più lento) |
| `FAISS_HNSW_EF_CONSTRUCTION` | `40` | Più basso = insert più veloci |
| `HNSW_EF_CONSTRUCTION` | `40` | Usato solo per ChromaDB |
| `PARALLEL_EMBED_WORKERS` | `1` | Thread embed (tenere 1, MPS non multi-processa bene) |
| `QUEUE_MAXSIZE` | `4` | Code graduate tra stadi (backpressure) |
| `STORAGE_FLUSH_SIZE` | `5000` | Flush nel vector store ogni N chunk |
| `SAVE_INTERVAL` | `5` | Persiste processed_titles ogni N flush |

## Performance (M1 Mac, 1000 articoli, ~15.6k chunk)

| Configurazione | ~chunks/s | Tempo totale |
|-----------|:---:|:---:|
| FAISS + `paraphrase-MiniLM-L3-v2` | **~1160** | **~13s** |
| ChromaDB + `paraphrase-MiniLM-L3-v2` | ~592 | ~26s |
| ChromaDB + `all-MiniLM-L6-v2` (precedente default) | ~288 | ~54s |

## Project structure

```
├── config.py         # LM Studio URL, chunk params, flag embedding
├── embeddings.py     # LMStudioEmbeddings + LocalEmbeddings (MPS)
├── ingest.py         # Extract articles from Wikipedia dump
├── index.py          # Pipeline parallela: producer → embed → store thread
├── rag.py            # LangChain RAG chain
├── app.py            # Streamlit UI
├── articles.jsonl    # Extracted articles (generated)
└── vector_store/     # FAISS index + sqlite metadata (generated)
    ├── faiss_index.bin
    └── metadata.sqlite
```
