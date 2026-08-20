from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class KnowledgeLayer(str, Enum):
    IDENTITY = "identity"
    EXPERIMENTAL = "experimental"
    SEQUENCE = "sequence"
    STRUCTURE = "structure"
    ATOM_CHEMISTRY = "atom_chemistry"
    EVOLUTIONARY = "evolutionary"
    FUNCTIONAL = "functional"
    MODEL = "model"
    LITERATURE = "literature"
    AGENT = "agent"
    PROVENANCE = "provenance"


class Modality(str, Enum):
    TABULAR = "tabular"
    SEQUENCE = "sequence"
    STRUCTURE_3D = "structure_3d"
    ATOMIC = "atomic"
    MSA = "msa"
    TEXT = "text"
    EMBEDDING = "embedding"
    TIME_SERIES = "time_series"
    ONTOLOGY = "ontology"


_SEMANTIC_ID_LOCK = threading.RLock()
_SEMANTIC_ID_COLLISIONS: dict[str, dict[str, str]] = {}


def stable_record_id(prefix: str, *parts: Any) -> str:
    """Return a deterministic readable ID without a content-hash contract."""

    def token(value: Any) -> str:
        if isinstance(value, (list, tuple, set)):
            raw = "-".join(token(item) for item in value)
        elif isinstance(value, dict):
            raw = "-".join(
                f"{token(key)}-{token(item)}" for key, item in sorted(value.items())
            )
        else:
            raw = str(value)
        cleaned = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-").upper() or "ITEM"
        if len(cleaned) <= 18:
            return cleaned
        return f"{cleaned[:7]}-{cleaned[-7:]}-L{len(cleaned):03d}"

    label = token(prefix)[:12]
    semantic_parts = "-".join(token(item) for item in parts)
    base = f"{label}:{semantic_parts or 'ITEM'}"
    canonical = repr(parts)
    with _SEMANTIC_ID_LOCK:
        bucket = _SEMANTIC_ID_COLLISIONS.setdefault(base, {})
        if canonical in bucket:
            return bucket[canonical]
        identifier = base if not bucket else f"{base}-N{len(bucket) + 1:02d}"
        bucket[canonical] = identifier
        return identifier


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    entity_type: str
    layer: KnowledgeLayer
    modalities: frozenset[Modality] = frozenset()
    properties: dict[str, Any] = field(default_factory=dict)
    source_ids: tuple[str, ...] = ()
    source_group: str = "unknown"
    confidence: float = 1.0
    valid_from_round: int | None = None
    valid_to_round: int | None = None


@dataclass(frozen=True)
class RelationRecord:
    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    layer: KnowledgeLayer
    modalities: frozenset[Modality] = frozenset()
    properties: dict[str, Any] = field(default_factory=dict)
    source_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    source_group: str = "unknown"
    confidence: float = 1.0
    context_id: str | None = None
    valid_from_round: int | None = None
    valid_to_round: int | None = None


@dataclass(frozen=True)
class KnowledgeBatch:
    adapter_name: str
    entities: tuple[EntityRecord, ...] = ()
    relations: tuple[RelationRecord, ...] = ()


@dataclass(frozen=True)
class BuildContext:
    run_id: str
    round_id: int
    protein_id: str
    assay_id: str | None = None
    condition_id: str | None = None
    resources: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")


@dataclass(frozen=True)
class KnowledgeGraphSnapshot:
    entities: tuple[EntityRecord, ...]
    relations: tuple[RelationRecord, ...]
