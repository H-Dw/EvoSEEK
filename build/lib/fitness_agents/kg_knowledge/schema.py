from __future__ import annotations

import hashlib
import json
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


def stable_record_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


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
