"""
retriever.py — FAISS-backed document retriever (one per domain).

Indexes are lazily loaded on first search call.
"""

import json
import pathlib
from typing import List, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = pathlib.Path(__file__).parent.parent / "data" / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


class Retriever:
    def __init__(self, domain: str):
        self.domain = domain
        self._index: Optional[faiss.Index] = None
        self._meta: Optional[List[Dict]] = None

    def _load(self):
        if self._index is not None:
            return
        faiss_path = INDEX_DIR / f"{self.domain}.faiss"
        meta_path = INDEX_DIR / f"{self.domain}_meta.json"
        if not faiss_path.exists():
            raise FileNotFoundError(
                f"Index missing: {faiss_path}\n"
                "Run 'python code/indexer.py' first."
            )
        self._index = faiss.read_index(str(faiss_path))
        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int = 5) -> str:
        """Return top-k relevant chunks as a formatted string."""
        try:
            self._load()
        except FileNotFoundError as e:
            return f"[Retrieval error: {e}]"

        model = _get_model()
        q_emb = model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)

        scores, indices = self._index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._meta[idx]
            results.append(
                f"[Source: {chunk.get('source', 'unknown')} | Relevance: {score:.3f}]\n"
                f"{chunk['text']}"
            )

        if not results:
            return f"No relevant documents found in the {self.domain} corpus."

        return "\n\n---\n\n".join(results)

    def search_all_domains(self, query: str, top_k: int = 3) -> str:
        """Search this domain and return results labelled with domain name."""
        result = self.search(query, top_k=top_k)
        return f"=== {self.domain.upper()} CORPUS ===\n{result}"


# Singleton retrievers — shared across the process
hackerrank_retriever = Retriever("hackerrank")
claude_retriever = Retriever("claude")
visa_retriever = Retriever("visa")
