#!/usr/bin/env python3
"""Chunk articles, embed, and store in ChromaDB.

Pipeline (parallel):
  producer thread  ->  read JSONL + chunk + skip-already-embedded
  embed worker(s)  ->  embed_documents() (runs on MPS / CPU / ONNX)
  store thread     ->  drain + upsert to ChromaDB every flush_size chunks

Resume: chunk ids are deterministic (hash(title) :: chunk_index), so re-runs
skip articles that are already present in the collection. `--reset` wipes first.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import argparse
import hashlib
import json
import os
import queue
import shutil
import threading
import time

import chromadb
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

import config
from embeddings import LocalEmbeddings, LMStudioEmbeddings


def stable_hash(title: str) -> str:
    """Deterministic chunk id prefix from the article title (<=128 chars after suffix)."""
    return hashlib.blake2b(title.encode("utf-8"), digest_size=16).hexdigest()


def build_embedder(batch_size=None):
    if config.USE_LOCAL_EMBEDDING:
        return LocalEmbeddings(
            use_fp16=config.EMBEDDING_USE_FP16,
            backend=getattr(config, "EMBEDDING_BACKEND", "torch"),
            encode_batch_size=batch_size or config.EMBEDDING_BATCH_SIZE,
        )
    return LMStudioEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.LM_STUDIO_URL)


def embed_worker(in_q, out_q, embedder, errors):
    """Pull batches from in_q, embed, push (ids, metas, docs, vectors) to out_q."""
    while True:
        batch = in_q.get()
        if batch is None:
            out_q.put(None)
            in_q.task_done()
            return
        try:
            texts = [b[3] for b in batch]
            vectors = embedder.embed_documents(texts)  # np.float32 2D
            ids = [b[0] for b in batch]
            metas = [{"title": b[1], "url": b[2]} for b in batch]
            docs = texts
            out_q.put((ids, metas, docs, vectors))
        except Exception as e:  # noqa: BLE001 - surface to main thread
            errors.append(e)
            out_q.put(None)
            in_q.task_done()
            return
        in_q.task_done()


def main():
    parser = argparse.ArgumentParser(description="Chunk articles, embed, store in ChromaDB")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing chroma_db and re-index from scratch")
    parser.add_argument("--encode-batch-size", type=int, default=None,
                        help=f"Override EMBEDDING_BATCH_SIZE (default {config.EMBEDDING_BATCH_SIZE})")
    parser.add_argument("--flush-size", type=int, default=None,
                        help=f"Override STORAGE_FLUSH_SIZE (default {config.STORAGE_FLUSH_SIZE})")
    args = parser.parse_args()

    batch_size = args.encode_batch_size or config.EMBEDDING_BATCH_SIZE
    flush_size = args.flush_size or config.STORAGE_FLUSH_SIZE

    processed_file = "processed_titles.json"
    if args.reset:
        if os.path.exists(config.CHROMA_DB_DIR):
            shutil.rmtree(config.CHROMA_DB_DIR)
            print(f"Deleted {config.CHROMA_DB_DIR}/")
        if os.path.exists(processed_file):
            os.remove(processed_file)
            print(f"Deleted {processed_file}")

    with open(config.ARTICLES_FILE) as f:
        n_articles = sum(1 for _ in f)
    print(f"{n_articles} articles found")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    print("Loading embedder...")
    embedder = build_embedder(batch_size=batch_size)

    client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    collection = client.get_or_create_collection("langchain")

    # Load existing chunk ids once: lets us skip already-embedded articles on resume.
    existing_ids = set()
    if not args.reset:
        fetched = collection.get(include=[])
        existing_ids = set(fetched.get("ids", []))
        print(f"Resume: {len(existing_ids)} chunks already in collection")

    # Load processed titles cache to avoid re-splitting on resume.
    processed_titles = set()
    if os.path.exists(processed_file):
        with open(processed_file) as f:
            processed_titles = set(json.load(f))
    print(f"Processed cache: {len(processed_titles)} articles")
    processed_lock = threading.Lock()

    qmax = getattr(config, "QUEUE_MAXSIZE", 4)
    batch_q = queue.Queue(maxsize=qmax)
    store_q = queue.Queue(maxsize=qmax * 2)  # ponytail: larger store_q since flush is async
    errors = []

    chunks_done = 0
    pbar = tqdm(total=n_articles, unit="art", desc="Indexing")
    start = time.time()

    # --- Producer thread: read + chunk + skip ---
    def run_producer():
        batch = []
        with open(config.ARTICLES_FILE) as f:
            for line in f:
                if errors:
                    break
                art = json.loads(line)
                title = art["title"]
                url = art["url"]
                title_hash = stable_hash(title)
                with processed_lock:
                    if title_hash in processed_titles:
                        pbar.update(1)
                        continue
                texts = splitter.split_text(art["text"])
                if texts and f"{title_hash}::{len(texts) - 1}" in existing_ids:
                    with processed_lock:
                        processed_titles.add(title_hash)
                    pbar.update(1)
                    continue
                for idx, t in enumerate(texts):
                    batch.append((f"{title_hash}::{idx}", title, url, t))
                    if len(batch) >= batch_size:
                        batch_q.put(batch)
                        batch = []
                pbar.update(1)
        if batch:
            batch_q.put(batch)
        batch_q.put(None)

    p_thread = threading.Thread(target=run_producer, name="producer")
    p_thread.start()

    # --- Embed worker thread(s) ---
    n_workers = max(1, getattr(config, "PARALLEL_EMBED_WORKERS", 1))
    workers = []
    for i in range(n_workers):
        t = threading.Thread(target=embed_worker, args=(batch_q, store_q, embedder, errors),
                             name=f"embed-{i}", daemon=True)
        t.start()
        workers.append(t)

    # --- Store thread: drain store_q, flush every flush_size chunks ---
    def run_store():
        nonlocal chunks_done
        MAX_UPSERT = 5000  # ponytail: ChromaDB max batch ~5461, stay safe

        def do_upsert(ids, metas, docs, vecs):
            all_vecs = np.vstack(vecs)
            for i in range(0, len(ids), MAX_UPSERT):
                end = i + MAX_UPSERT
                collection.upsert(
                    ids=ids[i:end],
                    embeddings=all_vecs[i:end],
                    metadatas=metas[i:end],
                    documents=docs[i:end],
                )
            with processed_lock:
                for cid in ids:
                    processed_titles.add(cid.split("::")[0])
            with open(processed_file, "w") as f:
                with processed_lock:
                    titles = sorted(processed_titles)
                json.dump(titles, f)

        s_ids, s_metas, s_docs, s_vecs = [], [], [], []
        sentinels_seen = 0
        while sentinels_seen < n_workers:
            item = store_q.get()
            if item is None:
                sentinels_seen += 1
                continue
            ids, metas, docs, vectors = item
            s_ids.extend(ids)
            s_metas.extend(metas)
            s_docs.extend(docs)
            s_vecs.append(vectors)
            chunks_done += len(ids)
            if len(s_ids) >= flush_size:
                do_upsert(s_ids, s_metas, s_docs, s_vecs)
                s_ids.clear(); s_metas.clear(); s_docs.clear(); s_vecs.clear()
        if s_ids:
            do_upsert(s_ids, s_metas, s_docs, s_vecs)

    s_thread = threading.Thread(target=run_store, name="store")
    s_thread.start()

    p_thread.join()
    for w in workers:
        w.join()
    s_thread.join()

    with open(processed_file, "w") as f:
        with processed_lock:
            titles = sorted(processed_titles)
        json.dump(titles, f)
    pbar.close()

    if errors:
        raise errors[0]

    total = time.time() - start
    rate = chunks_done / total if total else 0
    print(f"Done! {chunks_done} chunks embedded+stored in {total:.1f}s ({rate:.0f} chunks/s)")


if __name__ == "__main__":
    main()
