from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import quote


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
    """Return a process-order-independent semantic ID without a hash registry."""

    def canonicalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): canonicalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (set, frozenset)):
            normalized = [canonicalize(item) for item in value]
            return sorted(
                normalized,
                key=lambda item: json.dumps(
                    item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            )
        if isinstance(value, (list, tuple)):
            return [canonicalize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    label = re.sub(r"[^A-Za-z0-9_-]+", "-", str(prefix)).strip("-").upper()
    label = label or "RECORD"
    canonical = json.dumps(
        canonicalize(parts),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{label}:{quote(canonical, safe='-._~')}"


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
