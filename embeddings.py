from typing import List
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
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_fp16: bool = True):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        if use_fp16 and device == "mps":
            self.model.half()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, batch_size=512).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()
