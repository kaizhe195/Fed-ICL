"""Embedding index used for local retrieval.knn索引"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class EmbeddingIndex:
    """A small cosine-similarity index backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.examples: list[dict[str, Any]] = []
        self.text_key = "question"
        self.embeddings: np.ndarray | None = None
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "The 'sentence-transformers' package is required for kNN retrieval."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def build_index(self, examples: list[dict[str, Any]], text_key: str = "question") -> None:
        """Build the embedding matrix for examples."""
        self.examples = list(examples)
        self.text_key = text_key
        texts = [str(example.get(text_key, "")) for example in self.examples]
        self.embeddings = self._encode(texts)

    def search(self, query_text: str, top_k: int) -> list[tuple[dict[str, Any], float]]:
        """Return top-k examples and cosine similarities for one query."""
        if self.embeddings is None:
            raise RuntimeError("Embedding index has not been built or loaded.")
        if len(self.examples) == 0 or top_k <= 0:
            return []
        query_embedding = self._encode([query_text])[0]
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(-scores)[: min(top_k, len(self.examples))]
        return [(self.examples[int(index)], float(scores[int(index)])) for index in top_indices]

    def search_many(self, query_texts: list[str], top_k: int) -> list[list[tuple[dict[str, Any], float]]]:
        """Return top-k examples for each query."""
        return [self.search(query_text, top_k) for query_text in query_texts]

    def save_index(self, path: str | Path) -> None:
        """Persist examples and embeddings without serializing the model object."""
        if self.embeddings is None:
            raise RuntimeError("Cannot save an index before build_index.")
        payload = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "examples": self.examples,
            "text_key": self.text_key,
            "embeddings": self.embeddings,
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load_index(cls, path: str | Path) -> "EmbeddingIndex":
        """Load a persisted embedding index."""
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        index = cls(
            model_name=payload["model_name"],
            batch_size=payload.get("batch_size", 32),
            normalize_embeddings=payload.get("normalize_embeddings", True),
        )
        index.examples = payload["examples"]
        index.text_key = payload.get("text_key", "question")
        index.embeddings = payload["embeddings"]
        return index
