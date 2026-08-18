from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np


class SentenceTransformerEmbeddingBackend:
    def __init__(
        self,
        model_path: str | Path,
        *,
        model_id: str | None = None,
        revision: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
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
        self.model_id = model_id or path.name
        self.revision = revision or "local"
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        dimension_method = getattr(self.model, "get_embedding_dimension", None)
        dimension = (
            dimension_method()
            if callable(dimension_method)
            else self.model.get_sentence_embedding_dimension()
        )
        if dimension is None:
            raise RuntimeError("Could not determine local embedding dimension")
        self.dimension = int(dimension)
        self.max_input_tokens = int(self.model.max_seq_length)
        self.weight_hash = self._weight_hash(path)
        self.name = f"sentence-transformers:{self.model_id}@{self.revision}:{self.weight_hash[:16]}"
        self.fingerprint: dict[str, object] = {
            "backend": "sentence-transformers",
            "model_id": self.model_id,
            "revision": self.revision,
            "weight_hash": self.weight_hash,
            "dimension": self.dimension,
            "max_input_tokens": self.max_input_tokens,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "normalize_embeddings": True,
        }

    @staticmethod
    def _weight_hash(path: Path) -> str:
        candidates = sorted(
            item
            for item in path.rglob("*")
            if item.is_file()
            and (
                item.suffix.casefold() in {".safetensors", ".bin", ".json", ".txt", ".model"}
                or item.name in {"modules.json", "config_sentence_transformers.json"}
            )
        )
        if not candidates:
            raise RuntimeError(f"No model/config files were found under {path}")
        digest = hashlib.sha256()
        for item in candidates:
            relative = item.relative_to(path).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(str(item.stat().st_size).encode("ascii"))
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()

    def _encode(self, texts: Sequence[str], *, prefix: str) -> np.ndarray:
        prepared = [f"{prefix}{text}" for text in texts]
        values = self.model.encode(
            prepared,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, prefix=self.document_prefix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, prefix=self.query_prefix)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Backward-compatible document encoding for diagnostics and older callers."""

        return self.encode_documents(texts)

    def count_tokens(self, text: str, *, query: bool = False) -> int:
        prefix = self.query_prefix if query else self.document_prefix
        tokenizer = self.model.tokenizer
        encoded = tokenizer(
            f"{prefix}{text}",
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded.get("input_ids", ())
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return len(input_ids)


class SentenceTransformerRerankerBackend:
    score_kind = "raw_logit"

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_id: str | None = None,
        revision: str | None = None,
    ) -> None:
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
        self.model_id = model_id or path.name
        self.revision = revision or "local"
        self.weight_hash = SentenceTransformerEmbeddingBackend._weight_hash(path)
        self.name = (
            f"sentence-transformers-cross-encoder:{self.model_id}@{self.revision}:"
            f"{self.weight_hash[:16]}"
        )
        self.fingerprint = {
            "backend": "sentence-transformers-cross-encoder",
            "model_id": self.model_id,
            "revision": self.revision,
            "weight_hash": self.weight_hash,
        }

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        pairs = [(query, text) for text in texts]
        return np.asarray(self.model.predict(pairs, show_progress_bar=False), dtype=np.float32)
