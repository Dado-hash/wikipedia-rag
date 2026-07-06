import os
import pickle
import sqlite3
from functools import lru_cache
from typing import List

import faiss
import numpy as np
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import config
from embeddings import LocalEmbeddings, LMStudioEmbeddings


# ── Embeddings ───────────────────────────────────────────────────────────────

def _build_embeddings():
    if config.USE_LOCAL_EMBEDDING:
        return LocalEmbeddings(use_fp16=config.EMBEDDING_USE_FP16)
    return LMStudioEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.LM_STUDIO_URL)


# ── FAISS retriever (fixed rowid via IndexIDMap) ─────────────────────────────

def _build_faiss_retriever(embeddings):
    index = faiss.read_index(config.FAISS_INDEX_FILE)
    ef_search = getattr(config, "FAISS_HNSW_EF_SEARCH", 64)
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = ef_search
    is_idmap = isinstance(index, faiss.IndexIDMap)

    def retrieve(query, k=config.HYBRID_TOP_K):
        if isinstance(query, dict):
            query = query.get("input", query.get("query", ""))
        q = np.array([embeddings.embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(q)
        distances, indices = index.search(q, k)
        conn = sqlite3.connect(config.METADATA_DB_FILE)
        try:
            docs = []
            for idx in indices[0]:
                if idx < 0:
                    continue
                if is_idmap:
                    row = conn.execute(
                        "SELECT id, title, url, text FROM docs WHERE faiss_id = ?",
                        (int(idx),),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT id, title, url, text FROM docs WHERE rowid = ?",
                        (int(idx) + 1,),
                    ).fetchone()
                if row:
                    docs.append(Document(
                        page_content=row[3],
                        metadata={"id": row[0], "title": row[1], "url": row[2]},
                    ))
            return docs
        finally:
            conn.close()

    return RunnableLambda(retrieve)


# ── Chroma retriever (unchanged) ─────────────────────────────────────────────

def _build_chroma_retriever(embeddings):
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DB_DIR,
    )
    return vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})


# ── BM25 index loading ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_bm25_index():
    path = os.path.join(config.VECTOR_STORE_DIR, "bm25_index.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Cross-encoder (lazy loaded) ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_cross_encoder():
    return CrossEncoder(config.CROSS_ENCODER_MODEL)


# ── Advanced retrieval pipeline ──────────────────────────────────────────────

def _hyde_query(query: str, llm: BaseLanguageModel) -> str:
    """Hypothetical Document Embedding: generate a plausible answer,
    then use it as the retrieval query (embedding closer to relevant docs)."""
    prompt = ChatPromptTemplate.from_template(
        "Write a concise paragraph that answers the following question. "
        "Base it on facts you know.\n\nQuestion: {query}\n\nParagraph:"
    )
    chain = prompt | llm
    result = chain.invoke({"query": query})
    hyde_text = result.content if hasattr(result, "content") else str(result)
    return hyde_text


def _bm25_split(text: str) -> List[str]:
    """Tokenize text for BM25 scoring."""
    return text.split()


def _rrf_merge(dense_docs: List[Document], bm25_docs: List[Document],
               k: int = None) -> List[Document]:
    """Reciprocal Rank Fusion of two ranked lists."""
    if k is None:
        k = config.RRF_K
    seen = {}
    for rank, doc in enumerate(dense_docs):
        doc_id = doc.metadata.get("id", id(doc))
        seen[doc_id] = {"doc": doc, "rrf": 0.0}
        seen[doc_id]["dense_rank"] = rank + 1
    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.metadata.get("id", id(doc))
        if doc_id in seen:
            seen[doc_id]["bm25_rank"] = rank + 1
        else:
            seen[doc_id] = {"doc": doc, "dense_rank": None, "bm25_rank": rank + 1}
    for info in seen.values():
        score = 0.0
        if info.get("dense_rank"):
            score += 1.0 / (k + info["dense_rank"])
        if info.get("bm25_rank"):
            score += 1.0 / (k + info["bm25_rank"])
        info["rrf"] = score
    sorted_docs = sorted(seen.values(), key=lambda x: x["rrf"], reverse=True)
    return [item["doc"] for item in sorted_docs]


def _enrich_context(docs: List[Document], window: int = None) -> List[Document]:
    """Expand each retrieved doc with adjacent chunks from the same article."""
    if window is None:
        window = config.CONTEXT_WINDOW_SIZE
    if not docs:
        return docs
    conn = sqlite3.connect(config.METADATA_DB_FILE)
    try:
        seen = set()
        enriched = []
        for doc in docs:
            cid = doc.metadata.get("id", "")
            parts = cid.split("::")
            if len(parts) != 2:
                if cid not in seen:
                    seen.add(cid)
                    enriched.append(doc)
                continue
            article_hash, chunk_idx = parts[0], int(parts[1])
            rows = conn.execute(
                "SELECT id, title, url, text FROM docs WHERE id LIKE ? ORDER BY id",
                (f"{article_hash}::%",),
            ).fetchall()
            for row in rows:
                row_id = row[0]
                row_parts = row_id.split("::")
                if len(row_parts) != 2:
                    continue
                row_idx = int(row_parts[1])
                if abs(row_idx - chunk_idx) <= window:
                    if row_id not in seen:
                        seen.add(row_id)
                        enriched.append(Document(
                            page_content=row[3],
                            metadata={"id": row[0], "title": row[1], "url": row[2]},
                        ))
        return enriched
    finally:
        conn.close()


def _rerank_docs(query: str, docs: List[Document], top_k: int = None) -> List[Document]:
    """Re-rank documents with a cross-encoder."""
    if top_k is None:
        top_k = config.CROSS_ENCODER_TOP_K
    cross_encoder = _load_cross_encoder()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# ── Build retriever ──────────────────────────────────────────────────────────

def build_retriever(llm: BaseLanguageModel = None):
    """Build a retriever that may include HyDE, Hybrid Search, and Re-ranking."""
    embeddings = _build_embeddings()

    if config.VECTOR_STORE == "chroma":
        return _build_chroma_retriever(embeddings)

    # Build the dense-only retriever (RunnableLambda).
    dense_retriever = _build_faiss_retriever(embeddings)

    # Load BM25 index (if exists).
    bm25_index = _load_bm25_index() if config.ENABLE_HYBRID_SEARCH else None

    # Load DB for BM25 doc-text lookup.
    bm25_texts = None
    if bm25_index is not None:
        conn = sqlite3.connect(config.METADATA_DB_FILE)
        rows = conn.execute(
            "SELECT id, title, url, text FROM docs ORDER BY faiss_id"
        ).fetchall()
        conn.close()
        bm25_texts = rows  # list of (id, title, url, text)

    # Wrap everything in a single RunnableLambda.
    def retrieve(query):
        if isinstance(query, dict):
            raw_query = query.get("input", query.get("query", ""))
        else:
            raw_query = query

        search_query = raw_query

        # 1. HyDE: replace query with a hypothetical document embedding.
        if config.ENABLE_HYDE and llm is not None:
            hyde_doc = _hyde_query(raw_query, llm)
            search_query = hyde_doc

        # 2. Dense retrieval (k is set inside the retriever closure).
        dense_results = dense_retriever.invoke(search_query)

        # 3. Hybrid: fuse dense + BM25 via RRF.
        if bm25_index is not None and bm25_texts is not None:
            bm25_scores = bm25_index.get_scores(_bm25_split(raw_query))
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:config.HYBRID_TOP_K]
            bm25_results = []
            for pos in top_bm25_indices:
                if pos < len(bm25_texts):
                    row = bm25_texts[pos]
                    bm25_results.append(Document(
                        page_content=row[3],
                        metadata={"id": row[0], "title": row[1], "url": row[2]},
                    ))
            all_results = _rrf_merge(dense_results, bm25_results)
        else:
            all_results = dense_results

        # 4. Re-ranking with cross-encoder.
        if config.ENABLE_RERANKING and len(all_results) > 1:
            all_results = _rerank_docs(raw_query, all_results)

        # 5. Context Enrichment: add adjacent chunks for coherence.
        if config.ENABLE_CONTEXT_ENRICHMENT:
            all_results = _enrich_context(all_results)

        # Trim to TOP_K when re-ranking isn't reducing the list.
        if len(all_results) > config.TOP_K and not config.ENABLE_RERANKING:
            all_results = all_results[:config.TOP_K]

        return all_results

    return RunnableLambda(retrieve)


# ── RAG chain ────────────────────────────────────────────────────────────────

def build_rag_chain(retriever=None, model_name=None):
    if model_name is None:
        model_name = config.DEFAULT_CHAT_MODEL

    llm = ChatOpenAI(
        model=model_name,
        base_url=config.LM_STUDIO_URL,
        api_key="not-needed",
        temperature=0.3,
    )

    if retriever is None:
        retriever = build_retriever(llm=llm)

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant that answers questions based on "
        "the provided Wikipedia articles.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n\n"
        "Answer concisely and cite the article titles you used as sources."
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)
