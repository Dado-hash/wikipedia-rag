from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
import requests
import torch
from sentence_transformers import SentenceTransformer


class LMStudioEmbeddings(Embeddings):
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.url = f"{base_url}/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(self.url, json={"model": self.model, "input": texts})
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        resp = requests.post(self.url, json={"model": self.model, "input": text})
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


class LocalEmbeddings(Embeddings):
    """Sentence-transformers embeddings on MPS/CPU.

    `embed_documents` returns a contiguous `np.float32` 2D array (one row per
    text) so callers can stack/flush them to the vector store without
    per-element Python conversion.  `embed_query` returns a plain list for
    query-time compatibility with LangChain/Chroma.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2",
                 use_fp16: bool = True, encode_batch_size: int = 2048):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        if use_fp16 and device == "mps":
            self.model.half()
        self._encode_batch_size = encode_batch_size

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        # encode() already returns an np.ndarray; cast to float32 (cheap, vectorized)
        # instead of .tolist() which would build ~800k Python floats per 2048-batch.
        emb = self.model.encode(
            texts, show_progress_bar=False,
            batch_size=self._encode_batch_size,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def embed_query(self, text: str) -> List[float]:
        emb = self.model.encode(text, show_progress_bar=False)
        return emb.astype(np.float32).tolist()
