from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .schema import KnowledgeGraphSnapshot


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    record_id: str
    message: str


class KnowledgeValidator(Protocol):
    name: str

    def validate(self, snapshot: KnowledgeGraphSnapshot) -> tuple[ValidationIssue, ...]: ...


class CoreSchemaValidator:
    name = "core_schema"

    def validate(self, snapshot: KnowledgeGraphSnapshot) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        entity_ids: set[str] = set()
        for entity in snapshot.entities:
            if not entity.entity_id:
                issues.append(ValidationIssue("error", "empty_entity_id", "", "Entity ID is empty"))
            if entity.entity_id in entity_ids:
                issues.append(
                    ValidationIssue(
                        "error", "duplicate_entity_id", entity.entity_id, "Entity ID is duplicated"
                    )
                )
            entity_ids.add(entity.entity_id)
            if not 0.0 <= entity.confidence <= 1.0:
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_confidence",
                        entity.entity_id,
                        "Entity confidence must be in [0, 1]",
                    )
                )
            if (
                entity.valid_to_round is not None
                and entity.valid_from_round is not None
                and entity.valid_to_round < entity.valid_from_round
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_validity_interval",
                        entity.entity_id,
                        "valid_to_round precedes valid_from_round",
                    )
                )

        relation_ids: set[str] = set()
        for relation in snapshot.relations:
            if relation.relation_id in relation_ids:
                issues.append(
                    ValidationIssue(
                        "error",
                        "duplicate_relation_id",
                        relation.relation_id,
                        "Relation ID is duplicated",
                    )
                )
            relation_ids.add(relation.relation_id)
            for endpoint_name, endpoint_id in (
                ("subject", relation.subject_id),
                ("object", relation.object_id),
            ):
                if endpoint_id not in entity_ids:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "dangling_relation",
                            relation.relation_id,
                            f"{endpoint_name} endpoint {endpoint_id!r} does not exist",
                        )
                    )
            if not 0.0 <= relation.confidence <= 1.0:
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_confidence",
                        relation.relation_id,
                        "Relation confidence must be in [0, 1]",
                    )
                )
            if (
                relation.valid_to_round is not None
                and relation.valid_from_round is not None
                and relation.valid_to_round < relation.valid_from_round
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_validity_interval",
                        relation.relation_id,
                        "valid_to_round precedes valid_from_round",
                    )
                )
            if not relation.source_ids:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "missing_provenance",
                        relation.relation_id,
                        "Relation has no source IDs",
                    )
                )
        return tuple(issues)
