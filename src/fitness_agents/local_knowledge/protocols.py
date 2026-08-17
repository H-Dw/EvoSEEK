from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .contracts import ParsedDocument, RetrievalRequest, RetrievalResult


class ParserBackend(Protocol):
    name: str

    def supports(self, path: Path) -> bool: ...

    def parse(self, path: Path) -> ParsedDocument: ...


class EmbeddingBackend(Protocol):
    name: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class RerankerBackend(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray: ...


class KnowledgeSource(Protocol):
    name: str

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...
