from __future__ import annotations

from dataclasses import dataclass, replace

from fitness_agents.plugin_registry import PluginRegistry

from .ablation import KnowledgeAblationConfig
from .adapters import KnowledgeAdapter
from .fusion import FusionPolicy, ProvenanceAwareFusion
from .normalization import KnowledgeNormalizer
from .schema import (
    BuildContext,
    EntityRecord,
    KnowledgeBatch,
    KnowledgeGraphSnapshot,
    RelationRecord,
)
from .sinks import KnowledgeGraphSink
from .validation import CoreSchemaValidator, KnowledgeValidator, ValidationIssue


@dataclass(frozen=True)
class BuildReport:
    adapter_counts: dict[str, dict[str, int]]
    skipped_adapters: tuple[str, ...]
    filtered_entities: int
    filtered_relations: int
    dropped_dangling_relations: int
    validation_issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True)
class BuildResult:
    snapshot: KnowledgeGraphSnapshot
    report: BuildReport


class KnowledgeGraphBuilder:
    """BioCypher-style adapter pipeline with independent ablation switches."""

    def __init__(
        self,
        adapters: PluginRegistry[KnowledgeAdapter],
        *,
        config: KnowledgeAblationConfig | None = None,
        normalizers: tuple[KnowledgeNormalizer, ...] = (),
        fusion: FusionPolicy | None = None,
        validators: tuple[KnowledgeValidator, ...] = (),
        sinks: tuple[KnowledgeGraphSink, ...] = (),
        strict: bool = True,
    ) -> None:
        self.adapters = adapters
        self.config = config or KnowledgeAblationConfig()
        self.normalizers = normalizers
        self.fusion = fusion or ProvenanceAwareFusion()
        self.validators = validators or (CoreSchemaValidator(),)
        self.sinks = sinks
        self.strict = strict

    def _filter_entity(self, entity: EntityRecord) -> EntityRecord | None:
        if (
            self.config.enabled_layers is not None
            and entity.layer not in self.config.enabled_layers
        ):
            return None
        if (
            self.config.enabled_entity_types is not None
            and entity.entity_type not in self.config.enabled_entity_types
        ):
            return None
        if self.config.enabled_modalities is None or not entity.modalities:
            return entity
        modalities = entity.modalities.intersection(self.config.enabled_modalities)
        return replace(entity, modalities=frozenset(modalities)) if modalities else None

    def _filter_relation(self, relation: RelationRecord) -> RelationRecord | None:
        if (
            self.config.enabled_layers is not None
            and relation.layer not in self.config.enabled_layers
        ):
            return None
        if (
            self.config.enabled_relation_types is not None
            and relation.predicate not in self.config.enabled_relation_types
        ):
            return None
        if self.config.enabled_modalities is None or not relation.modalities:
            return relation
        modalities = relation.modalities.intersection(self.config.enabled_modalities)
        return replace(relation, modalities=frozenset(modalities)) if modalities else None

    def build(self, context: BuildContext) -> BuildResult:
        batches: list[KnowledgeBatch] = []
        counts: dict[str, dict[str, int]] = {}
        skipped: list[str] = []
        filtered_entities = 0
        filtered_relations = 0

        for name in self.adapters.names():
            if not self.config.adapter_enabled(name):
                skipped.append(name)
                continue
            batch = self.adapters.get(name).extract(context)
            for normalizer in self.normalizers:
                batch = normalizer.normalize(batch)
            kept_entities = tuple(
                item
                for entity in batch.entities
                if (item := self._filter_entity(entity)) is not None
            )
            kept_relations = tuple(
                item
                for relation in batch.relations
                if (item := self._filter_relation(relation)) is not None
            )
            filtered_entities += len(batch.entities) - len(kept_entities)
            filtered_relations += len(batch.relations) - len(kept_relations)
            counts[name] = {
                "input_entities": len(batch.entities),
                "input_relations": len(batch.relations),
                "kept_entities": len(kept_entities),
                "kept_relations": len(kept_relations),
            }
            batches.append(KnowledgeBatch(name, kept_entities, kept_relations))

        snapshot = self.fusion.fuse(batches)
        entity_ids = {entity.entity_id for entity in snapshot.entities}
        safe_relations = tuple(
            relation
            for relation in snapshot.relations
            if relation.subject_id in entity_ids and relation.object_id in entity_ids
        )
        dropped_dangling = len(snapshot.relations) - len(safe_relations)
        snapshot = KnowledgeGraphSnapshot(snapshot.entities, safe_relations)

        issues = tuple(
            issue for validator in self.validators for issue in validator.validate(snapshot)
        )
        errors = [issue for issue in issues if issue.severity == "error"]
        if self.strict and errors:
            summary = "; ".join(f"{item.code}:{item.record_id}" for item in errors[:5])
            raise ValueError(f"Knowledge graph validation failed: {summary}")
        for sink in self.sinks:
            sink.write(snapshot)
        return BuildResult(
            snapshot,
            BuildReport(
                adapter_counts=counts,
                skipped_adapters=tuple(skipped),
                filtered_entities=filtered_entities,
                filtered_relations=filtered_relations,
                dropped_dangling_relations=dropped_dangling,
                validation_issues=issues,
            ),
        )
