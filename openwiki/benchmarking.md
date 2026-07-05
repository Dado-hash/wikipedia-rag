# Benchmarking — `benchmark.py`

Pick the fastest embedding batch size for your hardware **before** committing
to a multi-hour reindex.

## What it does

1. Samples N real chunks from `articles.jsonl` (same splitter as `index.py`).
2. For each `batch_size`, warms up once (model load + kernel compile), then
   times `embed_documents()` over the full sample, keeping the best of 2
   rounds.
3. Prints a `chunks/s` table and names the winner with the exact `config.py`
   line to set.

Does **not** touch the vector store — pure embedding throughput.

## CLI

```bash
python benchmark.py                        # 1000 chunks, batches [512, 1024, 2048]
python benchmark.py --n 2000 --batch-sizes 512 1024 2048 4096
python benchmark.py --model sentence-transformers/all-MiniLM-L6-v2
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--n` | `1000` | Number of chunks to sample |
| `--batch-sizes` | `512 1024 2048` | Batch sizes to test |
| `--model` | `config.EMBEDDING_MODEL_NAME` | Embedding model to benchmark |

## Reading results

Output looks like:

```
model                                         device   batch        secs     chunks/s
---------------------------------------------------------------------------
sentence-transformers/paraphrase-MiniLM-L3-v2 mps      512          0.85         1172
sentence-transformers/paraphrase-MiniLM-L3-v2 mps      1024         0.82         1217
sentence-transformers/paraphrase-MiniLM-L3-v2 mps      2048         0.83         1202

Fastest: batch_size=1024 -> 1217 chunks/s
Set in config.py: EMBEDDING_BATCH_SIZE = 1024
```

The "fastest" line is a hint, not a mandate. If two configs are within ~5%,
prefer the larger batch (fewer Python loop iterations and store flushes) as
long as it doesn't OOM.

## Known numbers (M1 Mac)

| Model | Vector store | ~chunks/s | ~s/15.6k chunks |
|-------|--------------|-----------|-----------------|
| paraphrase-MiniLM-L3-v2 | FAISS | ~1160 | ~13 |
| paraphrase-MiniLM-L3-v2 | ChromaDB | ~592 | ~26 |
| all-MiniLM-L6-v2 | ChromaDB | ~288 | ~54 |

The default (L3 + FAISS) is ~4× faster than the previous default (L6 + ChromaDB)
on the same dataset.

## When to re-benchmark

- New hardware (different Mac, CUDA GPU, cloud box).
- After swapping the embedding model (different model = different optimal batch).
- After a `sentence-transformers` major version bump — kernel changes move the
  numbers.

Then reindex with `--reset` if the model or dimensions changed.
