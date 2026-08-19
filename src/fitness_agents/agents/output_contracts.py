from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from fitness_agents.agents.output_guards import UnknownEvidenceIdsError
from fitness_agents.contracts.schemas import Hypothesis, ReThinkReflection

HYPOTHESIS_TEXT_MAX = 400
RETHINK_TEXT_MAX = 400
EVIDENCE_ID_MAX = 12


def _non_empty_text(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("text values must not be empty")
    return cleaned


NonEmptyText = Annotated[str, AfterValidator(_non_empty_text)]
HypothesisText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=HYPOTHESIS_TEXT_MAX),
]
ReThinkText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=RETHINK_TEXT_MAX),
]
CANONICAL_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
_UNSET = object()
_VARIANT_ID_PREFIX = "sha256:"


def visible_evidence_ids(
    evidence_ids: Sequence[str],
    allowed_evidence_ids: frozenset[str] | None,
    *,
    on_unknown: Literal["error", "strip"] = "error",
) -> tuple[str, ...]:
    """Keep cited evidence IDs that the role can see; drop variant hashes.

    Remote scientists often copy observation `sha256:` identifiers into
    `evidence_ids`. Those are variant IDs, not evidence IDs, and must not fail
    the whole hypothesis when the rest of the contract is valid.
    """

    if allowed_evidence_ids is None:
        return tuple(evidence_ids)
    kept: list[str] = []
    unknown: list[str] = []
    for item in evidence_ids:
        if item in allowed_evidence_ids:
            kept.append(item)
        elif str(item).startswith(_VARIANT_ID_PREFIX):
            continue
        else:
            unknown.append(item)
    if unknown and on_unknown == "error":
        raise UnknownEvidenceIdsError(unknown, allowed_evidence_ids)
    return tuple(kept)


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


class ChannelContributionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    channel: Literal["physchem", "conservation", "structure"]
    sub_hypothesis_id: NonEmptyText
    claim: HypothesisText
    evidence_ids: Annotated[list[NonEmptyText], Field(max_length=EVIDENCE_ID_MAX)]
    uncertainty: HypothesisText


class HypothesisExplanationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    summary: HypothesisText
    channel_contributions: Annotated[
        list[ChannelContributionOutput], Field(min_length=1, max_length=3)
    ]
    conflicts: Annotated[list[dict[str, Any]], Field(max_length=12)]
    limitations: Annotated[list[HypothesisText], Field(min_length=1, max_length=8)]


class HypothesisOutput(BaseModel):
    """Strict model-facing contract converted to the existing Hypothesis dataclass."""

    model_config = ConfigDict(extra="forbid", strict=True)

    hypothesis_id: NonEmptyText
    statement: HypothesisText
    preferred_residues: dict[str, list[NonEmptyText]]
    evidence_ids: Annotated[list[NonEmptyText], Field(max_length=EVIDENCE_ID_MAX)]
    expected_outcome: HypothesisText
    falsification_criterion: HypothesisText
    parent_hypothesis_id: NonEmptyText | None
    explanation: HypothesisExplanationOutput | None = None

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
        allowed_positions: tuple[int, ...] | None = None,
        max_positions: int | None = None,
        on_unknown_evidence: Literal["error", "strip"] = "error",
    ) -> Hypothesis:
        hypothesis_id = self.hypothesis_id
        if expected_hypothesis_id is not None:
            # CampaignRunner owns this identifier; do not fail the round on a copy error.
            hypothesis_id = expected_hypothesis_id
        parent_hypothesis_id = self.parent_hypothesis_id
        if expected_parent_hypothesis_id is not _UNSET:
            parent_hypothesis_id = expected_parent_hypothesis_id
        if parent_hypothesis_id == hypothesis_id:
            raise ValueError("parent_hypothesis_id must differ from hypothesis_id")
        evidence_ids = visible_evidence_ids(
            self.evidence_ids,
            allowed_evidence_ids,
            on_unknown=on_unknown_evidence,
        )
        preferred = PreferredResiduesOutput(residues=self.preferred_residues).residues
        if expected_positions is not None:
            expected = {str(item) for item in expected_positions}
            actual = set(preferred)
            if actual != expected:
                raise ValueError(
                    "preferred_residues position mismatch; "
                    f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
                )
        if allowed_positions is not None:
            allowed = {str(item) for item in allowed_positions}
            unexpected = sorted(set(preferred).difference(allowed))
            if unexpected:
                raise ValueError(
                    f"preferred_residues contains positions outside design space: {unexpected}"
                )
        if max_positions is not None and len(preferred) > max_positions:
            raise ValueError(
                f"preferred_residues exceeds max_positions={max_positions}"
            )
        return Hypothesis(
            hypothesis_id=hypothesis_id,
            statement=self.statement,
            preferred_residues={
                int(site): tuple(residues) for site, residues in preferred.items()
            },
            evidence_ids=evidence_ids,
            expected_outcome=self.expected_outcome,
            falsification_criterion=self.falsification_criterion,
            parent_hypothesis_id=parent_hypothesis_id,
            explanation=(
                self.explanation.model_dump(mode="json")
                if self.explanation is not None
                else None
            ),
        )


class ReThinkItemOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    variant_id: NonEmptyText
    verdict: Literal["support", "conflict", "mixed", "inconclusive"]
    summary: ReThinkText
    positive_findings: list[NonEmptyText]
    negative_findings: list[NonEmptyText]
    revised_reason: ReThinkText
    next_round_advice: ReThinkText


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
    allowed_positions: tuple[int, ...] | None = None,
    max_positions: int | None = None,
    on_unknown_evidence: Literal["error", "strip"] = "error",
) -> dict[str, Any]:
    model = HypothesisOutput.model_validate(payload)
    try:
        hypothesis = model.to_hypothesis(
            expected_hypothesis_id=expected_hypothesis_id,
            expected_parent_hypothesis_id=expected_parent_hypothesis_id,
            allowed_evidence_ids=allowed_evidence_ids,
            expected_positions=expected_positions,
            allowed_positions=allowed_positions,
            max_positions=max_positions,
            on_unknown_evidence=on_unknown_evidence,
        )
    except UnknownEvidenceIdsError as error:
        stripped = model.to_hypothesis(
            expected_hypothesis_id=expected_hypothesis_id,
            expected_parent_hypothesis_id=expected_parent_hypothesis_id,
            allowed_evidence_ids=allowed_evidence_ids,
            expected_positions=expected_positions,
            allowed_positions=allowed_positions,
            max_positions=max_positions,
            on_unknown_evidence="strip",
        )
        dumped = model.model_dump(mode="json", by_alias=True)
        dumped["hypothesis_id"] = stripped.hypothesis_id
        dumped["parent_hypothesis_id"] = stripped.parent_hypothesis_id
        dumped["evidence_ids"] = list(stripped.evidence_ids)
        error.stripped_payload = dumped
        raise
    dumped = model.model_dump(mode="json", by_alias=True)
    dumped["hypothesis_id"] = hypothesis.hypothesis_id
    dumped["parent_hypothesis_id"] = hypothesis.parent_hypothesis_id
    dumped["evidence_ids"] = list(hypothesis.evidence_ids)
    return dumped


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
