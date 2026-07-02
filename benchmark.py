#!/usr/bin/env python3
"""Benchmark embedding backends and batch sizes to pick the fastest config.

Samples real chunks from articles.jsonl, then times `embed_documents()` across:
  - backends:   torch (MPS/CPU), onnx (CPU)  [onnx only if installed]
  - batch sizes: from --batch-sizes (default 64,128,256,512)

Prints a chunks/s table. Does NOT write to ChromaDB.

Usage:
    python benchmark.py                       # sample 1000 chunks, default batches
    python benchmark.py --n 2000 --batch-sizes 32 64 128 256 512
    python benchmark.py --skip-onnx           # torch-only run
"""
import argparse
import time

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config
from embeddings import LocalEmbeddings


def sample_chunks(n):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    with open(config.ARTICLES_FILE) as f:
        for line in f:
            if len(chunks) >= n:
                break
            art = __import__("json").loads(line)
            chunks.extend(splitter.split_text(art["text"]))
    return chunks[:n]


def time_backend(backend, batch_size, texts, rounds=2):
    emb = LocalEmbeddings(
        backend=backend,
        use_fp16=config.EMBEDDING_USE_FP16,
        encode_batch_size=batch_size,
    )
    device = "mps" if backend == "torch" else "cpu"
    # warmup (first run pays model load / kernel compile)
    emb.embed_documents(texts[:batch_size])
    best = float("inf")
    for _ in range(rounds):
        t0 = time.time()
        emb.embed_documents(texts)
        best = min(best, time.time() - t0)
    rate = len(texts) / best
    return best, rate, device


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding backends & batch sizes")
    parser.add_argument("--n", type=int, default=1000, help="number of chunks to sample")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--skip-onnx", action="store_true", help="do not test the ONNX backend")
    args = parser.parse_args()

    print(f"Sampling {args.n} chunks from {config.ARTICLES_FILE} ...")
    texts = sample_chunks(args.n)
    print(f"Sampled {len(texts)} chunks (chunk_size={config.CHUNK_SIZE})\n")

    backends = ["torch"]
    if not args.skip_onnx:
        try:
            import onnxruntime  # noqa: F401
            backends.append("onnx")
        except ImportError:
            print("onnxruntime not installed -> skipping ONNX (pip install onnxruntime optimum)\n")

    print(f"{'backend':<10} {'device':<8} {'batch':<8} {'secs':>8} {'chunks/s':>12}")
    print("-" * 50)
    results = {}
    for backend in backends:
        for bs in args.batch_sizes:
            try:
                secs, rate, device = time_backend(backend, bs, texts)
            except Exception as e:  # noqa: BLE001
                print(f"{backend:<10} {'-':<8} {bs:<8} {'ERROR':>8} {e}")
                continue
            print(f"{backend:<10} {device:<8} {bs:<8} {secs:>8.2f} {rate:>12.0f}")
            results[(backend, bs)] = rate

    if results:
        (best_key, best_rate) = max(results.items(), key=lambda kv: kv[1])
        print(f"\nFastest: backend={best_key[0]} batch_size={best_key[1]} -> {best_rate:.0f} chunks/s")
        print(f"Set in config.py: EMBEDDING_BACKEND = \"{best_key[0]}\", EMBEDDING_BATCH_SIZE = {best_key[1]}")


if __name__ == "__main__":
    main()
