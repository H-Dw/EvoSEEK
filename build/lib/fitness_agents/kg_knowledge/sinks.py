from __future__ import annotations

from typing import Protocol

from .schema import KnowledgeGraphSnapshot


class KnowledgeGraphSink(Protocol):
    name: str

    def write(self, snapshot: KnowledgeGraphSnapshot) -> None: ...


class InMemoryGraphSink:
    name = "memory"

    def __init__(self) -> None:
        self.snapshot: KnowledgeGraphSnapshot | None = None

    def write(self, snapshot: KnowledgeGraphSnapshot) -> None:
        self.snapshot = snapshot
