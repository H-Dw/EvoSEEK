from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Protocol, TypeVar

from .schema import (
    EntityRecord,
    KnowledgeBatch,
    KnowledgeGraphSnapshot,
    RelationRecord,
    stable_record_id,
)


class FusionPolicy(Protocol):
    name: str

    def fuse(self, batches: Iterable[KnowledgeBatch]) -> KnowledgeGraphSnapshot: ...


RecordT = TypeVar("RecordT", EntityRecord, RelationRecord)


def _source_aware_confidence(records: Iterable[RecordT]) -> float:
    """Use max within a source family, noisy-or across independent families."""

    by_group: dict[str, float] = defaultdict(float)
    for record in records:
        confidence = min(1.0, max(0.0, float(record.confidence)))
        by_group[record.source_group] = max(by_group[record.source_group], confidence)
    probability_none = math.prod(1.0 - confidence for confidence in by_group.values())
    return 1.0 - probability_none


def _merge_properties(records: Iterable[RecordT]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    conflicts: dict[str, list[Any]] = {}
    for record in records:
        for key, value in record.properties.items():
            if key not in merged:
                merged[key] = value
            elif merged[key] != value:
                values = conflicts.setdefault(key, [merged[key]])
                if value not in values:
                    values.append(value)
    if conflicts:
        merged["_conflicts"] = conflicts
    return merged


class ProvenanceAwareFusion:
    """Deduplicate records without double-counting correlated evidence sources."""

    name = "provenance_aware"

    def fuse(self, batches: Iterable[KnowledgeBatch]) -> KnowledgeGraphSnapshot:
        entity_groups: dict[str, list[EntityRecord]] = defaultdict(list)
        relation_groups: dict[tuple[Any, ...], list[RelationRecord]] = defaultdict(list)
        for batch in batches:
            for entity in batch.entities:
                entity_groups[entity.entity_id].append(entity)
            for relation in batch.relations:
                key = (
                    relation.subject_id,
                    relation.predicate,
                    relation.object_id,
                    relation.context_id,
                    relation.valid_from_round,
                    relation.valid_to_round,
                )
                relation_groups[key].append(relation)

        entities: list[EntityRecord] = []
        for entity_id in sorted(entity_groups):
            records = entity_groups[entity_id]
            first = records[0]
            entity_types = {item.entity_type for item in records}
            layers = {item.layer for item in records}
            if len(entity_types) != 1 or len(layers) != 1:
                raise ValueError(f"Conflicting schema assignments for entity {entity_id!r}")
            entities.append(
                replace(
                    first,
                    modalities=frozenset().union(*(item.modalities for item in records)),
                    properties=_merge_properties(records),
                    source_ids=tuple(
                        sorted({source for item in records for source in item.source_ids})
                    ),
                    source_group="fused" if len(records) > 1 else first.source_group,
                    confidence=_source_aware_confidence(records),
                )
            )

        relations: list[RelationRecord] = []
        for key in sorted(relation_groups, key=lambda item: tuple(str(value) for value in item)):
            records = relation_groups[key]
            first = records[0]
            layers = {item.layer for item in records}
            if len(layers) != 1:
                raise ValueError(f"Conflicting layers for relation {key!r}")
            relation_id = stable_record_id("relation-fused", *key)
            relations.append(
                replace(
                    first,
                    relation_id=relation_id,
                    modalities=frozenset().union(*(item.modalities for item in records)),
                    properties=_merge_properties(records),
                    source_ids=tuple(
                        sorted({source for item in records for source in item.source_ids})
                    ),
                    evidence_ids=tuple(
                        sorted({evidence for item in records for evidence in item.evidence_ids})
                    ),
                    source_group="fused" if len(records) > 1 else first.source_group,
                    confidence=_source_aware_confidence(records),
                )
            )
        return KnowledgeGraphSnapshot(tuple(entities), tuple(relations))
