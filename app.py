import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import streamlit as st
import config
from rag import build_retriever, build_rag_chain

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

if 'chain' not in st.session_state or st.session_state.get('model_name') != model_name:
    with st.spinner(f'Caricamento modello: {model_name}...'):
        retriever = get_retriever()
        st.session_state.chain = build_rag_chain(retriever=retriever, model_name=model_name)
        st.session_state.model_name = model_name

query = st.text_input('Ask a question:', placeholder='e.g. Who was Abraham Lincoln?')

if query:
    with st.spinner('Searching Wikipedia...'):
        result = st.session_state.chain.invoke({'input': query})

    st.markdown('### Answer')
    st.write(result['answer'])

    with st.expander(f'Sources ({len(result["context"])})', expanded=False):
        for i, doc in enumerate(result['context']):
            title = doc.metadata.get('title', '?')
            url = doc.metadata.get('url', '')
            st.markdown(f'**{i+1}. [{title}]({url})**')
            st.write(doc.page_content[:300] + '...')
            if i < len(result['context']) - 1:
                st.divider()
