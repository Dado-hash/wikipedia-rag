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
```

- **Chunking**: `RecursiveCharacterTextSplitter` (800 caratteri, overlap 100)
- **Embedding**: sentence-transformers (`all-MiniLM-L6-v2`) via MPS (GPU Apple Silicon) — ~300-500 chunk/s
- **Storage**: ChromaDB bulk insert (two-phase: prima embed, poi store)

Usa `--reset` per ricreare l'indice da capo.

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
| `CHUNK_SIZE` | `800` | Caratteri per chunk |
| `TOP_K` | `5` | Documenti recuperati per query |

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
├── index.py         # Two-phase: embed all, then bulk store in ChromaDB
├── rag.py           # LangChain RAG chain
├── app.py           # Streamlit UI
├── articles.jsonl   # Extracted articles (generated)
└── chroma_db/       # Vector store (generated)
```
