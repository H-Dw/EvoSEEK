from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from .schema import KnowledgeBatch, stable_record_id


class KnowledgeNormalizer(Protocol):
    name: str

    def normalize(self, batch: KnowledgeBatch) -> KnowledgeBatch: ...


class IdentityNormalizer:
    name = "identity"

    def normalize(self, batch: KnowledgeBatch) -> KnowledgeBatch:
        return batch


class AliasNormalizer:
    """Resolve adapter-local identifiers before provenance-aware fusion."""

    name = "aliases"

    def __init__(self, aliases: dict[str, str]) -> None:
        self.aliases = dict(aliases)

    def _canonical(self, entity_id: str) -> str:
        visited: set[str] = set()
        current = entity_id
        while current in self.aliases:
            if current in visited:
                raise ValueError(f"Alias cycle contains {current!r}")
            visited.add(current)
            current = self.aliases[current]
        return current

    def normalize(self, batch: KnowledgeBatch) -> KnowledgeBatch:
        entities = []
        for entity in batch.entities:
            canonical = self._canonical(entity.entity_id)
            properties = dict(entity.properties)
            if canonical != entity.entity_id:
                properties["canonicalized_from"] = entity.entity_id
            entities.append(replace(entity, entity_id=canonical, properties=properties))

        relations = []
        for relation in batch.relations:
            subject_id = self._canonical(relation.subject_id)
            object_id = self._canonical(relation.object_id)
            relation_id = stable_record_id(
                "rel",
                batch.adapter_name,
                subject_id,
                relation.predicate,
                object_id,
                relation.context_id,
                relation.valid_from_round,
                relation.valid_to_round,
            )
            relations.append(
                replace(
                    relation,
                    relation_id=relation_id,
                    subject_id=subject_id,
                    object_id=object_id,
                )
            )
        return KnowledgeBatch(batch.adapter_name, tuple(entities), tuple(relations))
