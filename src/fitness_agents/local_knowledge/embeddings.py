from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


class SentenceTransformerEmbeddingBackend:
    name = "sentence-transformers-local"

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Local embedding model does not exist: {path}")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for dense local retrieval; install the rag extra"
            ) from error
        self.model_path = path
        self.model = SentenceTransformer(str(path), local_files_only=True)
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Could not determine local embedding dimension")
        self.dimension = int(dimension)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)


class SentenceTransformerRerankerBackend:
    name = "sentence-transformers-cross-encoder-local"

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Local reranker model does not exist: {path}")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for local reranking; install the rag extra"
            ) from error
        self.model_path = path
        self.model = CrossEncoder(str(path), local_files_only=True)

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        pairs = [(query, text) for text in texts]
        return np.asarray(self.model.predict(pairs, show_progress_bar=False), dtype=np.float32)
