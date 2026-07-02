#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import json
import os
import shutil
import time
import uuid

import chromadb
import config
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings import LocalEmbeddings, LMStudioEmbeddings


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Chunk articles, embed, store in ChromaDB')
    parser.add_argument('--reset', action='store_true', help='Delete existing chroma_db and re-index')
    args = parser.parse_args()

    if args.reset and os.path.exists(config.CHROMA_DB_DIR):
        shutil.rmtree(config.CHROMA_DB_DIR)
        print(f'Deleted {config.CHROMA_DB_DIR}/')

    print('Loading articles...')
    with open(config.ARTICLES_FILE) as f:
        articles = [json.loads(line) for line in f]
    print(f'  {len(articles)} articles loaded')

    print('Chunking...')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=['\n\n', '\n', '. ', ' ', ''],
    )

    chunks = []
    for art in tqdm(articles, desc='Chunking', unit='article'):
        texts = splitter.split_text(art['text'])
        for t in texts:
            chunks.append({
                'title': art['title'],
                'url': art['url'],
                'text': t,
            })

    print(f'  {len(chunks)} chunks created')

    print(f'Loading embeddings...')
    if config.USE_LOCAL_EMBEDDING:
        embedder = LocalEmbeddings(use_fp16=config.EMBEDDING_USE_FP16)
    else:
        embedder = LMStudioEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.LM_STUDIO_URL,
        )

    texts = [c['text'] for c in chunks]

    print(f'Phase 1: Embedding {len(chunks)} chunks...')
    start = time.time()

    pbar = tqdm(total=len(chunks), unit='chunk', desc='Embedding')
    batch_size = config.EMBEDDING_BATCH_SIZE
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = texts[i:i + batch_size]
        batch_vecs = embedder.embed_documents(batch)
        vectors.extend(batch_vecs)
        pbar.update(len(batch))
        elapsed = time.time() - start
        done = i + len(batch)
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (len(chunks) - done) / rate if rate > 0 else 0
        pbar.set_postfix(rate=f'{rate:.0f} ch/s', eta=f'{remaining:.0f}s')
    pbar.close()

    embed_elapsed = time.time() - start
    print(f'  Done in {embed_elapsed:.1f}s ({len(chunks)/embed_elapsed:.0f} chunks/s)')

    print(f'Phase 2: Bulk storing in ChromaDB...')
    store_start = time.time()

    client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    collection = client.get_or_create_collection("langchain")

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{'title': c['title'], 'url': c['url']} for c in chunks]

    collection.add(
        ids=ids,
        embeddings=vectors,
        metadatas=metadatas,
        documents=texts,
    )

    store_elapsed = time.time() - store_start
    total = time.time() - start
    print(f'  Storage done in {store_elapsed:.1f}s ({len(chunks)/store_elapsed:.0f} chunks/s)')
    print(f'\nDone! {len(chunks)} chunks indexed in {total:.1f}s total')
    print(f'  Embed: {embed_elapsed:.1f}s | Store: {store_elapsed:.1f}s')


if __name__ == '__main__':
    main()
