#!/usr/bin/env python3
"""Benchmark embedding batch sizes to pick the fastest config.

Samples real chunks from articles.jsonl, then times `embed_documents()` across
the configured embedding model and a set of batch sizes.

Usage:
    python benchmark.py                       # 1000 chunks, default batch sizes
    python benchmark.py --n 2000 --batch-sizes 512 1024 2048 4096
    python benchmark.py --model sentence-transformers/all-MiniLM-L6-v2
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


def time_backend(model_name, batch_size, texts, rounds=2):
    emb = LocalEmbeddings(
        model_name=model_name,
        use_fp16=config.EMBEDDING_USE_FP16,
        encode_batch_size=batch_size,
    )
    device = "mps" if emb.model.device.type == "mps" else "cpu"
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
    parser = argparse.ArgumentParser(description="Benchmark embedding batch sizes")
    parser.add_argument("--n", type=int, default=1000, help="number of chunks to sample")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--model", default=config.EMBEDDING_MODEL_NAME, help="embedding model to benchmark")
    args = parser.parse_args()

    print(f"Sampling {args.n} chunks from {config.ARTICLES_FILE} ...")
    texts = sample_chunks(args.n)
    print(f"Sampled {len(texts)} chunks (chunk_size={config.CHUNK_SIZE}, model={args.model})\n")

    print(f"{'model':<45s} {'device':<8s} {'batch':<8s} {'secs':>8s} {'chunks/s':>12s}")
    print("-" * 75)
    results = {}
    for bs in args.batch_sizes:
        try:
            secs, rate, device = time_backend(args.model, bs, texts)
        except Exception as e:  # noqa: BLE001
            print(f"{args.model:<45s} {'-':<8s} {bs:<8d} {'ERROR':>8s} {e}")
            continue
        print(f"{args.model:<45s} {device:<8s} {bs:<8d} {secs:>8.2f} {rate:>12.0f}")
        results[bs] = rate

    if results:
        best_bs = max(results, key=results.get)
        best_rate = results[best_bs]
        print(f"\nFastest: batch_size={best_bs} -> {best_rate:.0f} chunks/s")
        print(f"Set in config.py: EMBEDDING_BATCH_SIZE = {best_bs}")


if __name__ == "__main__":
    main()
