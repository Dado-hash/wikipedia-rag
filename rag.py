import config
from embeddings import LocalEmbeddings, LMStudioEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def build_retriever():
    if config.USE_LOCAL_EMBEDDING:
        embeddings = LocalEmbeddings(use_fp16=config.EMBEDDING_USE_FP16)
    else:
        embeddings = LMStudioEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.LM_STUDIO_URL,
        )

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DB_DIR,
    )

    return vectorstore.as_retriever(
        search_kwargs={'k': config.TOP_K},
    )


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
