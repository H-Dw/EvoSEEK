from __future__ import annotations

import hashlib
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

from fitness_agents.contracts.schemas import Hypothesis, ReThinkReflection


def _non_empty_text(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("text values must not be empty")
    return cleaned


NonEmptyText = Annotated[str, AfterValidator(_non_empty_text)]
CANONICAL_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
_UNSET = object()


class PreferredResiduesOutput(BaseModel):
    """Dynamic task-position map; exact keys are checked against CampaignRunner context."""

    model_config = ConfigDict(extra="forbid", strict=True)

    residues: dict[str, list[NonEmptyText]]

    @field_validator("residues")
    @classmethod
    def validate_residue_map(
        cls, values: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        if not values:
            raise ValueError("preferred_residues must not be empty")
        output: dict[str, list[str]] = {}
        for site, residues in values.items():
            try:
                normalized_site = str(int(site))
            except ValueError as error:
                raise ValueError(f"preferred_residues site is not an integer: {site!r}") from error
            if not residues:
                raise ValueError("preferred residue arrays must not be empty")
            cleaned = [residue.upper() for residue in residues]
            invalid = [residue for residue in cleaned if residue not in CANONICAL_RESIDUES]
            if invalid:
                raise ValueError(
                    f"preferred_residues contains non-canonical residues: {invalid}"
                )
            if len(cleaned) != len(set(cleaned)):
                raise ValueError("preferred_residues contains duplicate residues")
            output[normalized_site] = cleaned
        return output


class HypothesisOutput(BaseModel):
    """Strict model-facing contract converted to the existing Hypothesis dataclass."""

    model_config = ConfigDict(extra="forbid", strict=True)

    hypothesis_id: NonEmptyText
    statement: NonEmptyText
    preferred_residues: dict[str, list[NonEmptyText]]
    evidence_ids: list[NonEmptyText]
    expected_outcome: NonEmptyText
    falsification_criterion: NonEmptyText
    parent_hypothesis_id: NonEmptyText | None

    @model_validator(mode="after")
    def validate_identifier_relationships(self) -> HypothesisOutput:
        if self.parent_hypothesis_id == self.hypothesis_id:
            raise ValueError("parent_hypothesis_id must differ from hypothesis_id")
        return self

    def to_hypothesis(
        self,
        *,
        expected_hypothesis_id: str | None = None,
        expected_parent_hypothesis_id: str | None | object = _UNSET,
        allowed_evidence_ids: frozenset[str] | None = None,
        expected_positions: tuple[int, ...] | None = None,
    ) -> Hypothesis:
        if expected_hypothesis_id is not None and self.hypothesis_id != expected_hypothesis_id:
            raise ValueError(
                f"hypothesis_id must equal CampaignRunner ID {expected_hypothesis_id!r}"
            )
        if (
            expected_parent_hypothesis_id is not _UNSET
            and self.parent_hypothesis_id != expected_parent_hypothesis_id
        ):
            raise ValueError("parent_hypothesis_id must match the current CampaignRunner context")
        if allowed_evidence_ids is not None:
            unknown = sorted(set(self.evidence_ids).difference(allowed_evidence_ids))
            if unknown:
                raise ValueError(f"evidence_ids contains identifiers not visible to the role: {unknown}")
        preferred = PreferredResiduesOutput(residues=self.preferred_residues).residues
        if expected_positions is not None:
            expected = {str(item) for item in expected_positions}
            actual = set(preferred)
            if actual != expected:
                raise ValueError(
                    "preferred_residues position mismatch; "
                    f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
                )
        return Hypothesis(
            hypothesis_id=self.hypothesis_id,
            statement=self.statement,
            preferred_residues={
                int(site): tuple(residues) for site, residues in preferred.items()
            },
            evidence_ids=tuple(self.evidence_ids),
            expected_outcome=self.expected_outcome,
            falsification_criterion=self.falsification_criterion,
            parent_hypothesis_id=self.parent_hypothesis_id,
        )


class ReThinkItemOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    variant_id: NonEmptyText
    verdict: Literal["support", "conflict", "mixed", "inconclusive"]
    summary: NonEmptyText
    positive_findings: list[NonEmptyText]
    negative_findings: list[NonEmptyText]
    revised_reason: NonEmptyText
    next_round_advice: NonEmptyText


class ReThinkOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reflections: list[ReThinkItemOutput]

    @model_validator(mode="after")
    def validate_unique_variants(self) -> ReThinkOutput:
        variant_ids = [item.variant_id for item in self.reflections]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("ReThink reflections must have unique variant_id values")
        return self

    def to_reflections(
        self, *, run_id: str, round_id: int, provider: str
    ) -> tuple[ReThinkReflection, ...]:
        output: list[ReThinkReflection] = []
        for item in self.reflections:
            digest = hashlib.sha256(
                f"{run_id}|{round_id}|{item.variant_id}".encode()
            ).hexdigest()[:16]
            output.append(
                ReThinkReflection(
                    reflection_id=f"rethink:{digest}",
                    variant_id=item.variant_id,
                    round_id=round_id,
                    verdict=item.verdict,
                    summary=item.summary,
                    positive_findings=tuple(item.positive_findings),
                    negative_findings=tuple(item.negative_findings),
                    revised_reason=item.revised_reason,
                    next_round_advice=item.next_round_advice,
                    provider=provider,
                )
            )
        return tuple(output)


def validate_hypothesis_payload(
    payload: dict[str, Any],
    *,
    expected_hypothesis_id: str | None = None,
    expected_parent_hypothesis_id: str | None | object = _UNSET,
    allowed_evidence_ids: frozenset[str] | None = None,
    expected_positions: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    model = HypothesisOutput.model_validate(payload)
    model.to_hypothesis(
        expected_hypothesis_id=expected_hypothesis_id,
        expected_parent_hypothesis_id=expected_parent_hypothesis_id,
        allowed_evidence_ids=allowed_evidence_ids,
        expected_positions=expected_positions,
    )
    return model.model_dump(mode="json", by_alias=True)


def validate_rethink_payload(
    payload: dict[str, Any], *, expected_variant_ids: frozenset[str] | None = None
) -> dict[str, Any]:
    model = ReThinkOutput.model_validate(payload)
    if expected_variant_ids is not None:
        actual = {item.variant_id for item in model.reflections}
        missing = sorted(expected_variant_ids.difference(actual))
        unexpected = sorted(actual.difference(expected_variant_ids))
        if missing or unexpected:
            raise ValueError(
                f"ReThink variant coverage mismatch; missing={missing}, unexpected={unexpected}"
            )
    return model.model_dump(mode="json")
