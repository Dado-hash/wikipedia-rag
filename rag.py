import sqlite3

import faiss
import numpy as np
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

import config
from embeddings import LocalEmbeddings, LMStudioEmbeddings


def _build_embeddings():
    if config.USE_LOCAL_EMBEDDING:
        return LocalEmbeddings(use_fp16=config.EMBEDDING_USE_FP16)
    return LMStudioEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.LM_STUDIO_URL)


def _build_chroma_retriever():
    embeddings = _build_embeddings()
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DB_DIR,
    )
    return vectorstore.as_retriever(search_kwargs={'k': config.TOP_K})


def _build_faiss_retriever():
    embeddings = _build_embeddings()
    index = faiss.read_index(config.FAISS_INDEX_FILE)
    ef_search = getattr(config, "FAISS_HNSW_EF_SEARCH", 64)
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = ef_search

    def retrieve(query):
        if isinstance(query, dict):
            query = query.get("input", query.get("query", ""))
        q = np.array([embeddings.embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(q)
        distances, indices = index.search(q, config.TOP_K)
        conn = sqlite3.connect(config.METADATA_DB_FILE)
        try:
            docs = []
            for idx in indices[0]:
                if idx < 0:
                    continue
                # FAISS returns 0-based positions; sqlite rowids are 1-based.
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


def build_retriever():
    if config.VECTOR_STORE == "chroma":
        return _build_chroma_retriever()
    return _build_faiss_retriever()


def build_rag_chain(retriever=None, model_name=None):
    if retriever is None:
        retriever = build_retriever()
    if model_name is None:
        model_name = config.DEFAULT_CHAT_MODEL

    llm = ChatOpenAI(
        model=model_name,
        base_url=config.LM_STUDIO_URL,
        api_key='not-needed',
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_template(
        'You are a helpful assistant that answers questions based on '
        'the provided Wikipedia articles.\n\n'
        'Context:\n{context}\n\n'
        'Question: {input}\n\n'
        'Answer concisely and cite the article titles you used as sources.'
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)
