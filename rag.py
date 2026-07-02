import config
from embeddings import LocalEmbeddings, LMStudioEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def build_rag_chain():
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

    retriever = vectorstore.as_retriever(
        search_kwargs={'k': config.TOP_K},
    )

    llm = ChatOpenAI(
        model=config.CHAT_MODEL,
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
