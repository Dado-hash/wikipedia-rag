import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import streamlit as st
from rag import build_rag_chain

st.set_page_config(page_title='Wikipedia RAG', layout='centered')
st.title('Wikipedia RAG')

if 'chain' not in st.session_state:
    with st.spinner('Loading RAG chain...'):
        st.session_state.chain = build_rag_chain()

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
