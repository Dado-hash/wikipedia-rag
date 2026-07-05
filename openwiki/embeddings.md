# Embeddings — `embeddings.py`

Two implementations of LangChain's `Embeddings` interface. Selected at
runtime by `config.USE_LOCAL_EMBEDDING` in both `index.py` (`build_embedder`)
and `rag.py` (`build_retriever`).

> **Critical:** index-time and query-time embeddings **must** use the same
> model. `articles` embedded with `all-MiniLM-L6-v2` (384-dim) cannot be
> queried with `paraphrase-MiniLM-L3-v2` if dimensions differ (both are
> 384-dim, but the internal weights are different). The default keeps both
> paths consistent via `config.EMBEDDING_MODEL_NAME`.

## `LocalEmbeddings` (default, recommended)

Wraps `sentence_transformers.SentenceTransformer(config.EMBEDDING_MODEL_NAME)`.

- **Device**: MPS if available, else CPU.
- **fp16**: on MPS when `EMBEDDING_USE_FP16=True`, halves memory with no
  measurable quality loss for these models.
- **Model**: `sentence-transformers/paraphrase-MiniLM-L3-v2` is the default.
  It is ~3× faster than `all-MiniLM-L6-v2` on Apple Silicon and supports a
  512-token context, so `CHUNK_SIZE=2000` English text no longer gets
  truncated. Both models produce 384-dimensional vectors.

### Return types matter

```python
embed_documents(texts) -> np.ndarray   # float32, 2D, contiguous
embed_query(text)      -> List[float]  # plain list
```

`embed_documents` returns a NumPy array intentionally: the store thread does
`np.vstack(vecs)` and flushes to the vector store. A `.tolist()` would build
~8M Python floats per 2048-batch for no benefit.
`embed_query` returns a list because LangChain/Chroma expect that at query
time.

## `LMStudioEmbeddings`

Thin HTTP client over LM Studio's `/v1/embeddings` endpoint. Used only when
`USE_LOCAL_EMBEDDING=False`. Kept as a fallback for environments without
`sentence-transformers`. Much slower than local MPS.

## Model

Default: `sentence-transformers/paraphrase-MiniLM-L3-v2` —
384-dimensional, ~3× faster than `all-MiniLM-L6-v2` on Apple Silicon.
To swap models, change `EMBEDDING_MODEL_NAME` and reindex from scratch.

## When to use which

| Situation | Use |
|-----------|-----|
| Apple Silicon Mac | `LocalEmbeddings` with the default L3 model |
| Need max quality | `all-MiniLM-L6-v2` or larger — slower |
| No `sentence-transformers` install | `LMStudioEmbeddings` |

See [benchmarking.md](benchmarking.md) for how to measure the tradeoff on
your hardware before committing to a reindex.
