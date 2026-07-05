# Quickstart

Wikipedia RAG: Retrieval-Augmented Generation over the English Wikipedia dump.
LangChain + Streamlit + LM Studio, with local sentence-transformers embeddings
running on Apple Silicon MPS (or CPU).

> Entry point for humans and agents. The other pages in `openwiki/` go deeper.

## What this project does

1. Extract articles from a Wikipedia XML dump (`ingest.py`).
2. Chunk, embed, and store them via a parallel pipeline (`index.py`).
3. Answer questions against that index with a local LLM via LM Studio (`app.py` + `rag.py`).

## Prerequisites

- Python 3.9+
- [LM Studio](https://lmstudio.ai/) running on `localhost:1234` with any OpenAI-compatible chat model loaded (default `google/gemma-4-12b-qat`)
- Apple Silicon recommended (MPS) — works on CPU too, slower
- The Wikipedia dump file `enwiki-latest-pages-articles.xml.bz2` in the repo root for step 1.
  Download from https://dumps.wikimedia.org/enwiki/latest/ if you don't have it.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run it (3 commands)

```bash
# 1. Extract articles from the dump -> articles.jsonl
python3 ingest.py --max-articles 5000

# 2. Chunk + embed + store in FAISS (default)
python3 index.py --reset

# 3. Launch the Streamlit UI
streamlit run app.py
```

Open http://localhost:8501, pick a chat model in the sidebar, ask a question.

## Fastest indexing config (M1 Mac)

```bash
python3 index.py --reset --encode-batch-size=2048 --flush-size=5000
```

~1160 chunks/s, ~13s for 1000 articles (~15.6k chunks). See
[benchmarking.md](benchmarking.md) and [indexing.md](indexing.md).

## Where to go next

| Page | Covers |
|------|--------|
| [architecture.md](architecture.md) | System overview, data flow, module map |
| [ingestion.md](ingestion.md) | `ingest.py` — dump parsing, wikitext cleaning |
| [indexing.md](indexing.md) | `index.py` — parallel producer→embed→store pipeline |
| [embeddings.md](embeddings.md) | `embeddings.py` — local vs LM Studio, model choices |
| [rag-and-ui.md](rag-and-ui.md) | `rag.py` + `app.py` — retrieval chain, Streamlit UI |
| [configuration.md](configuration.md) | `config.py` — every tunable parameter |
| [benchmarking.md](benchmarking.md) | `benchmark.py` — pick the fastest embedding config |

## Generated artifacts (gitignored)

- `articles.jsonl` — extracted articles from step 1
- `vector_store/` — FAISS index + sqlite metadata from step 2
- `chroma_db/` — optional ChromaDB store if `VECTOR_STORE="chroma"`
- `processed_titles.json` — resume cache for the indexer
- `enwiki-latest-pages-articles.xml.bz2` — the dump itself
