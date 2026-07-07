import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import streamlit as st
import config
from rag import build_retriever, build_streaming_chain

@st.cache_resource
def get_retriever():
    return build_retriever()

st.set_page_config(page_title='Wikipedia RAG', layout='centered')
st.title('Wikipedia RAG')

model_name = st.sidebar.selectbox('Modello', config.AVAILABLE_CHAT_MODELS)

st.sidebar.markdown('### Techniques')
badges = []
if config.ENABLE_HYDE:
    badges.append('🔄 HyDE')
if config.ENABLE_HYBRID_SEARCH:
    badges.append('⚡ Hybrid')
if config.ENABLE_RERANKING:
    badges.append('🎯 Re-rank')
if badges:
    st.sidebar.markdown(' '.join(f'`{b}`' for b in badges))
else:
    st.sidebar.markdown('`Dense only`')

query = st.text_input('Ask a question:', placeholder='e.g. Who was Abraham Lincoln?')

if query:
    retriever = get_retriever()
    with st.spinner('Searching Wikipedia...'):
        docs = retriever.invoke(query)

    streaming_chain = build_streaming_chain(model_name=model_name)

    st.markdown('### Answer')
    st.write_stream(
        chunk.content for chunk in streaming_chain.stream({"context": docs, "input": query})
    )

    with st.expander(f'Sources ({len(docs)})', expanded=False):
        for i, doc in enumerate(docs):
            title = doc.metadata.get('title', '?')
            url = doc.metadata.get('url', '')
            st.markdown(f'**{i+1}. [{title}]({url})**')
            st.write(doc.page_content[:300] + '...')
            if i < len(docs) - 1:
                st.divider()
