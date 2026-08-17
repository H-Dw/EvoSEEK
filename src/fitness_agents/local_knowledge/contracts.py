from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParsedDocument:
    document_id: str
    path: Path
    file_hash: str
    mime_type: str
    title: str
    text: str
    knowledge_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    section_path: tuple[str, ...]
    start_offset: int
    end_offset: int
    token_count: int
    source_group: str
    artifact_uri: str
    file_hash: str
    knowledge_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LeakagePolicyContext:
    enabled: bool
    policy_version: str
    protected_terms_hash: str
    generic_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalRequest:
    query_id: str
    round_id: int
    intent: str
    query: str
    anchors: tuple[str, ...] = ()
    top_k: int = 8
    token_budget: int = 5000
    filters: dict[str, Any] = field(default_factory=dict)
    policy_context: LeakagePolicyContext | None = None

    def __post_init__(self) -> None:
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        if not self.query_id or not self.intent:
            raise ValueError("RetrievalRequest requires query_id and intent")
        if self.top_k < 1 or self.token_budget < 1:
            raise ValueError("RetrievalRequest budgets must be positive")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    artifact_uri: str
    section_path: tuple[str, ...]
    start_offset: int
    end_offset: int
    source_group: str
    knowledge_type: str
    scores: dict[str, float]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class KnowledgeClaim:
    claim_id: str
    statement: str
    subject: str | None
    predicate: str | None
    object: str | None
    polarity: str
    applicability: dict[str, Any]
    confidence: float
    evidence_chunk_ids: tuple[str, ...]
    extraction_version: str = "retrieval-only:v1"

    def __post_init__(self) -> None:
        if self.polarity not in {"support", "contradict", "neutral", "unknown"}:
            raise ValueError("KnowledgeClaim polarity is invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("KnowledgeClaim confidence must be in [0, 1]")


@dataclass(frozen=True)
class RetrievalResult:
    query_id: str
    round_id: int
    original_query_hash: str
    sanitized_query: str
    policy_decision: dict[str, Any]
    chunks: tuple[RetrievedChunk, ...]
    claims: tuple[KnowledgeClaim, ...]
    warnings: tuple[str, ...]
    index_manifest_hash: str


@dataclass(frozen=True)
class IndexBuildReport:
    manifest_hash: str
    indexed_documents: int
    indexed_chunks: int
    unchanged_documents: int
    removed_documents: int
    quarantined_documents: int
    warnings: tuple[str, ...] = ()
