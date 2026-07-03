# Wikipedia RAG

Retrieval-Augmented Generation over the English Wikipedia dump, using LangChain + Streamlit + LM Studio.

## Prerequisites

- Python 3.9+
- [LM Studio](https://lmstudio.ai/) running on `localhost:1234`
  - **Chat model**: `google/gemma-4-12b-qat` (or any OpenAI-compatible model)
- sentence-transformers (per l'embedding locale, già in `requirements.txt`)

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
python3 index.py --encode-batch-size=1024 --flush-size=2000   # tuning
```

- **Chunking**: `RecursiveCharacterTextSplitter` (2000 caratteri, overlap 200)
- **Pipeline**: 3 thread in cascata — producer → embed worker → store thread (dedicato, upsert non blocca la pipeline)
- **Embedding**: sentence-transformers (`all-MiniLM-L6-v2`) via MPS (GPU Apple Silicon) — ~300-500 chunk/s
- **Storage**: ChromaDB upsert batch ogni `flush_size` chunk

CLI disponibili:
- `--reset`: ricrea l'indice da capo
- `--encode-batch-size N`: sovrascrive `EMBEDDING_BATCH_SIZE` (default 512)
- `--flush-size N`: sovrascrive `STORAGE_FLUSH_SIZE` (default 5000)

> **Tuning**: la configurazione più veloce finora è `--encode-batch-size=2048 --flush-size=10000`.
> ChromaDB ha un limite di ~5461 elementi per UPSERT, quindi flush anche più grandi vengono
> automaticamente suddivisi in blocchi da 5000 dal thread store.

### 3. Launch the UI

```bash
streamlit run app.py
```

Apri http://localhost:8501. Le query embedding usano lo stesso modello locale (compatibile con l'indice).

## Configurazione

Tutti i parametri in `config.py`:

| Parametro | Default | Note |
|-----------|---------|------|
| `LM_STUDIO_URL` | `http://localhost:1234/v1` | Solo per chat LLM |
| `CHAT_MODEL` | `google/gemma-4-12b-qat` | Modello per risposte |
| `USE_LOCAL_EMBEDDING` | `True` | Usa sentence-transformers (MPS) invece di LM Studio API |
| `EMBEDDING_BATCH_SIZE` | `512` | Batch per producer + SentenceTransformer encode |
| `EMBEDDING_BACKEND` | `"torch"` | `"torch"` o `"onnx"` (più veloce su CPU) |
| `EMBEDDING_USE_FP16` | `True` | Mezza precisione su MPS |
| `CHUNK_SIZE` | `2000` | Caratteri per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap tra chunk consecutivi |
| `TOP_K` | `5` | Documenti recuperati per query |
| `PARALLEL_EMBED_WORKERS` | `1` | Thread embed (tenere 1, MPS non multi-processa bene) |
| `QUEUE_MAXSIZE` | `4` | Code graduate tra stadi (backpressure) |
| `STORAGE_FLUSH_SIZE` | `5000` | Upsert in ChromaDB ogni N chunk |

## Performance (M1 Mac, 5000 articoli, ~194k chunk)

| Fase | LM Studio API | sentence-transformers (MPS) |
|------|:---:|:---:|
| Embedding | ~130 ch/s | **~310 ch/s** |
| Tempo totale | ~25 min | **~10 min** |
| Dump completo (est.) | ~26 giorni | **~10 giorni** |

## Project structure

```
├── config.py        # LM Studio URL, chunk params, flag embedding
├── embeddings.py    # LMStudioEmbeddings + LocalEmbeddings (MPS)
├── ingest.py        # Extract articles from Wikipedia dump
├── index.py         # Pipeline parallela: producer → embed → store thread
├── rag.py           # LangChain RAG chain
├── app.py           # Streamlit UI
├── articles.jsonl   # Extracted articles (generated)
└── chroma_db/       # Vector store (generated)
```
