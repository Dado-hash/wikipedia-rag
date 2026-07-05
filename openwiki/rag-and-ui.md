# RAG chain & UI — `rag.py` + `app.py`

## `rag.py`

Two builders. `app.py` calls both; `index.py` does not use this file.

### `build_retriever()` (`rag.py:13`)

Returns a retriever for the vector store selected by `config.VECTOR_STORE`:

- `"faiss"`: loads `config.FAISS_INDEX_FILE`, embeds the query, L2-normalizes
  it, runs a FAISS `search`, and maps the returned positions back to chunk
  metadata via `config.METADATA_DB_FILE`.
- `"chroma"`: constructs a LangChain `Chroma` vectorstore over
  `config.CHROMA_DB_DIR` with `vectorstore.as_retriever(search_kwargs={'k':
  config.TOP_K})`.

The embedding-function parity point from [embeddings.md](embeddings.md)
applies here: this is the query-time side. It must use the same model class
as was used at index time.

### `build_rag_chain(retriever, model_name)` (`rag.py:44`)

- `ChatOpenAI` pointed at `config.LM_STUDIO_URL` (`localhost:1234/v1`),
  `api_key='not-needed'`, `temperature=0.3`.
- A `ChatPromptTemplate` that stuffs `{context}` and `{input}`, instructing
  concise answers with source-title citations.
- `create_stuff_documents_chain(llm, prompt)` → combines retrieved docs into
  the prompt.
- `create_retrieval_chain(retriever, combine_docs_chain)` → the full chain.

`chain.invoke({'input': query})` returns `{'input', 'context', 'answer'}`,
where `context` is the list of retrieved `Document`s (with `title`/`url`
metadata carried from `ingest.py`).

## `app.py`

A 39-line Streamlit UI.

- `get_retriever()` is `@st.cache_resource`d — the FAISS index + embedding
  model load once per session, not per query (`app.py:8`).
- The chain is rebuilt **only** when the sidebar model selection changes
  (`app.py:17`). Switching models does NOT touch the retriever (same index,
  same embeddings) — only the chat LLM endpoint changes.
- Model dropdown comes from `config.AVAILABLE_CHAT_MODELS`.
- Output: the answer as markdown, plus a collapsible "Sources" expander
  listing each retrieved doc's title (linked to its Wikipedia URL) and a
  300-char preview.

## Prompt

```
You are a helpful assistant that answers questions based on the provided
Wikipedia articles.

Context:
{context}

Question: {input}

Answer concisely and cite the article titles you used as sources.
```

Temperature 0.3 keeps answers grounded; the citation instruction is what
makes the Sources panel line up with the answer text. Edit in `rag.py:42`.

## LM Studio contract

The app assumes LM Studio is already running with a chat model loaded. There
is no health check — a missing/unreachable LM Studio surfaces as a
`ChatOpenAI` connection error at query time. The model name in the dropdown
must match a model loaded in LM Studio; mismatches give a 404 from the API.
