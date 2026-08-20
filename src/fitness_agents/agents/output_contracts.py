from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
    field_validator,
    model_validator,
)

from fitness_agents.agents.output_guards import (
    SemanticOutputValidationError,
    UnknownEvidenceIdsError,
)
from fitness_agents.contracts.hypothesis_pipeline import SynthesisAbstention
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


def visible_evidence_ids(
    evidence_ids: Sequence[str],
    allowed_evidence_ids: frozenset[str] | None,
    *,
    on_unknown: Literal["error", "strip"] = "error",
) -> tuple[str, ...]:
    """Keep cited evidence IDs that the role can see."""

    if allowed_evidence_ids is None:
        return tuple(evidence_ids)
    kept: list[str] = []
    unknown: list[str] = []
    for item in evidence_ids:
        if item in allowed_evidence_ids:
            kept.append(item)
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
    analysis_id: NonEmptyText
    analysis_summary: HypothesisText
    evidence_ids: Annotated[list[NonEmptyText], Field(max_length=EVIDENCE_ID_MAX)]
    uncertainty: HypothesisText
    candidate_hypothesis_ids: Annotated[
        list[NonEmptyText], Field(max_length=4)
    ] = Field(default_factory=list)


class HypothesisExplanationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    summary: HypothesisText
    channel_contributions: Annotated[
        list[ChannelContributionOutput], Field(min_length=1, max_length=3)
    ]
    conflicts: Annotated[list[dict[str, Any]], Field(max_length=12)]
    limitations: Annotated[list[HypothesisText], Field(min_length=1, max_length=8)]


class FalsificationTemplateOutput(BaseModel):
    """The currently supported deterministic hypothesis-test compiler input."""

    model_config = ConfigDict(extra="forbid", strict=True)

    detector: Literal["batch_median_lift"]
    target_relation: Literal["selected_batch"]
    comparator_relation: Literal["pre_round_visible_observations"]
    operator: Literal["greater"]
    threshold_source: Literal["zero_lift"]
    min_observations: Literal["selected_batch_size"]
    missing_data_policy: Literal["INCONCLUSIVE"]
    reduction_policy: Literal["primary_contradiction_first_v1"]

    def render_description(self) -> str:
        return (
            "The selected batch median must exceed the preregistered pre-round "
            "visible-observation median; missing required observations yield INCONCLUSIVE."
        )


class HypothesisBodyOutput(BaseModel):
    """Compact model-facing hypothesis body.

    Hypothesis and parent identifiers are runtime-owned.  Scientific
    explanations are produced by the Critic, so neither belongs in the
    Scientist's generative contract.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_runtime_fields(cls, value: Any) -> Any:
        """Ignore fields that older prompts incorrectly assigned to the Scientist."""

        if not isinstance(value, dict):
            return value
        output = {
            key: item
            for key, item in value.items()
            if key not in {"hypothesis_id", "parent_hypothesis_id", "explanation"}
        }
        preferred = output.get("preferred_residues", {})
        output.setdefault("claim_modality", "directional_prior")
        output.setdefault(
            "preference_strength_by_position",
            {str(position): "soft" for position in preferred},
        )
        output.setdefault(
            "falsification_template",
            {
                "detector": "batch_median_lift",
                "target_relation": "selected_batch",
                "comparator_relation": "pre_round_visible_observations",
                "operator": "greater",
                "threshold_source": "zero_lift",
                "min_observations": "selected_batch_size",
                "missing_data_policy": "INCONCLUSIVE",
                "reduction_policy": "primary_contradiction_first_v1",
            },
        )
        return output

    statement: HypothesisText
    claim_modality: Literal[
        "directional_prior", "association", "mechanistic_hypothesis"
    ]
    preferred_residues: dict[str, list[NonEmptyText]]
    preference_strength_by_position: dict[
        str, Literal["soft", "exploratory"]
    ]
    evidence_ids: Annotated[list[NonEmptyText], Field(max_length=EVIDENCE_ID_MAX)]
    expected_outcome: HypothesisText
    falsification_criterion: HypothesisText
    hard_residue_constraints: dict[str, list[NonEmptyText]] = Field(
        default_factory=dict
    )
    falsification_template: FalsificationTemplateOutput

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
        if expected_hypothesis_id is None:
            raise ValueError("runtime must provide expected_hypothesis_id")
        hypothesis_id = expected_hypothesis_id
        parent_hypothesis_id = (
            None
            if expected_parent_hypothesis_id is _UNSET
            else expected_parent_hypothesis_id
        )
        if parent_hypothesis_id == hypothesis_id:
            raise ValueError("parent_hypothesis_id must differ from hypothesis_id")
        evidence_ids = visible_evidence_ids(
            self.evidence_ids,
            allowed_evidence_ids,
            on_unknown=on_unknown_evidence,
        )
        preferred = PreferredResiduesOutput(residues=self.preferred_residues).residues
        hard_constraints = (
            PreferredResiduesOutput(
                residues=self.hard_residue_constraints
            ).residues
            if self.hard_residue_constraints
            else {}
        )
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
            unexpected_hard = sorted(set(hard_constraints).difference(allowed))
            if unexpected_hard:
                raise ValueError(
                    "hard_residue_constraints contains positions outside design space: "
                    f"{unexpected_hard}"
                )
        if max_positions is not None and len(preferred) > max_positions:
            raise ValueError(
                f"preferred_residues exceeds max_positions={max_positions}"
            )
        if set(self.preference_strength_by_position) != set(preferred):
            raise ValueError(
                "preference_strength_by_position must exactly cover preferred_residues"
            )
        if not hard_constraints:
            hard_language = (
                "must ", "required", "forbidden", "only residue", "必须", "禁止", "不可变"
            )
            prose = f"{self.statement} {self.expected_outcome}".casefold()
            if any(marker in prose for marker in hard_language):
                raise ValueError(
                    "soft hypothesis prose cannot declare required or forbidden residues"
                )
        rendered_falsification = self.falsification_template.render_description()
        return Hypothesis(
            hypothesis_id=hypothesis_id,
            statement=self.statement,
            preferred_residues={
                int(site): tuple(residues) for site, residues in preferred.items()
            },
            evidence_ids=evidence_ids,
            expected_outcome=self.expected_outcome,
            falsification_criterion=rendered_falsification,
            parent_hypothesis_id=parent_hypothesis_id,
            hard_residue_constraints={
                int(site): tuple(residues)
                for site, residues in hard_constraints.items()
            },
            claim_modality=self.claim_modality,
            preference_strength_by_position={
                int(site): strength
                for site, strength in self.preference_strength_by_position.items()
            },
            falsification_template=self.falsification_template.model_dump(mode="json"),
        )


class HypothesisOutput(HypothesisBodyOutput):
    """Backward-compatible runtime envelope; not sent to remote models."""

    hypothesis_id: NonEmptyText
    parent_hypothesis_id: NonEmptyText | None

    @model_validator(mode="after")
    def validate_identifier_relationships(self) -> HypothesisOutput:
        if self.parent_hypothesis_id == self.hypothesis_id:
            raise ValueError("parent_hypothesis_id must differ from hypothesis_id")
        return self

    def to_hypothesis(self, **kwargs: Any) -> Hypothesis:
        kwargs.setdefault("expected_hypothesis_id", self.hypothesis_id)
        kwargs.setdefault("expected_parent_hypothesis_id", self.parent_hypothesis_id)
        return super().to_hypothesis(**kwargs)


class SynthesizedHypothesisOutput(HypothesisBodyOutput):
    outcome: Literal["SYNTHESIZED_HYPOTHESIS"] = "SYNTHESIZED_HYPOTHESIS"


class NoSupportedHypothesisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    outcome: Literal["NO_SUPPORTED_HYPOTHESIS"] = "NO_SUPPORTED_HYPOTHESIS"
    abstention_id: NonEmptyText
    reason: HypothesisText
    evidence_ids: Annotated[list[NonEmptyText], Field(max_length=EVIDENCE_ID_MAX)]
    unresolved_constraints: Annotated[list[HypothesisText], Field(max_length=8)]
    recommended_next_evidence: Annotated[list[HypothesisText], Field(max_length=8)]

    def to_abstention(
        self,
        *,
        allowed_evidence_ids: frozenset[str] | None,
    ) -> SynthesisAbstention:
        evidence_ids = visible_evidence_ids(
            self.evidence_ids, allowed_evidence_ids, on_unknown="error"
        )
        return SynthesisAbstention(
            abstention_id=self.abstention_id,
            reason=self.reason,
            evidence_ids=evidence_ids,
            unresolved_constraints=tuple(self.unresolved_constraints),
            recommended_next_evidence=tuple(self.recommended_next_evidence),
        )


MainSynthesisResult = Annotated[
    SynthesizedHypothesisOutput | NoSupportedHypothesisOutput,
    Field(discriminator="outcome"),
]


class MainSynthesisOutput(RootModel[MainSynthesisResult]):
    pass


def validate_main_synthesis_payload(
    payload: dict[str, Any],
    *,
    expected_hypothesis_id: str,
    expected_parent_hypothesis_id: str | None,
    allowed_evidence_ids: frozenset[str],
    expected_positions: tuple[int, ...] | None,
    allowed_positions: tuple[int, ...] | None,
    max_positions: int | None,
) -> dict[str, Any]:
    output = MainSynthesisOutput.model_validate(payload).root
    if isinstance(output, NoSupportedHypothesisOutput):
        abstention = output.to_abstention(allowed_evidence_ids=allowed_evidence_ids)
        dumped = output.model_dump(mode="json")
        dumped["evidence_ids"] = list(abstention.evidence_ids)
        return dumped
    try:
        hypothesis = output.to_hypothesis(
            expected_hypothesis_id=expected_hypothesis_id,
            expected_parent_hypothesis_id=expected_parent_hypothesis_id,
            allowed_evidence_ids=allowed_evidence_ids,
            expected_positions=expected_positions,
            allowed_positions=allowed_positions,
            max_positions=max_positions,
        )
    except ValueError as error:
        if "preferred_residues position mismatch" not in str(error):
            raise
        raise SemanticOutputValidationError(
            (
                f"{error}. With preference_policy=all_positions, either provide a non-empty, "
                "evidence-supported residue list for every mutable position or change outcome "
                "to NO_SUPPORTED_HYPOTHESIS; never use empty residue arrays."
            ),
            paths=("outcome", "preferred_residues"),
        ) from error
    dumped = output.model_dump(mode="json")
    dumped["evidence_ids"] = list(hypothesis.evidence_ids)
    return dumped


class ReThinkItemOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    variant_id: NonEmptyText
    candidate_relation: Literal["support", "conflict", "mixed", "inconclusive"]
    summary: ReThinkText
    positive_findings: list[NonEmptyText]
    negative_findings: list[NonEmptyText]
    revised_reason: ReThinkText
    next_round_advice: ReThinkText
    next_round_action: Literal[
        "retain_uncertainty_aware_exploration",
        "downweight_rationale",
        "test_matched_alternative",
        "collect_missing_measurement",
        "preserve_control",
        "no_change",
    ]


class ReThinkBatchAssessmentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    assessment_id: NonEmptyText | None
    status: Literal["SUPPORTED", "CONTRADICTED", "INCONCLUSIVE", "NOT_APPLICABLE"]
    commentary: ReThinkText
    next_round_advice: ReThinkText


class ReThinkOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reflections: list[ReThinkItemOutput]
    batch_assessment: ReThinkBatchAssessmentOutput

    @model_validator(mode="after")
    def validate_unique_variants(self) -> ReThinkOutput:
        variant_ids = [item.variant_id for item in self.reflections]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("ReThink reflections must have unique variant_id values")
        return self

    def to_reflections(
        self, *, run_id: str, round_id: int, provider: str, id_prefix: str = "R"
    ) -> tuple[ReThinkReflection, ...]:
        del run_id
        output: list[ReThinkReflection] = []
        for index, item in enumerate(self.reflections, start=1):
            output.append(
                ReThinkReflection(
                    reflection_id=f"{id_prefix}{round_id:02d}-{index:02d}",
                    variant_id=item.variant_id,
                    round_id=round_id,
                    verdict=item.candidate_relation,
                    summary=item.summary,
                    positive_findings=tuple(item.positive_findings),
                    negative_findings=tuple(item.negative_findings),
                    revised_reason=item.revised_reason,
                    next_round_advice=item.next_round_advice,
                    next_round_action=item.next_round_action,
                    provider=provider,
                    assessment_id=self.batch_assessment.assessment_id,
                    assessment_status=self.batch_assessment.status,
                    assessment_commentary=self.batch_assessment.commentary,
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
    model = HypothesisBodyOutput.model_validate(payload)
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
        dumped["evidence_ids"] = list(stripped.evidence_ids)
        error.stripped_payload = dumped
        raise
    dumped = model.model_dump(mode="json", by_alias=True)
    dumped["evidence_ids"] = list(hypothesis.evidence_ids)
    return dumped


def validate_rethink_payload(
    payload: dict[str, Any],
    *,
    expected_variant_ids: frozenset[str] | None = None,
    expected_assessment_id: str | None | object = _UNSET,
    expected_assessment_status: str | None = None,
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
    if expected_assessment_id is not _UNSET:
        if model.batch_assessment.assessment_id != expected_assessment_id:
            raise ValueError("ReThink batch assessment ID does not match runtime assessment")
        if (
            expected_assessment_status is not None
            and model.batch_assessment.status != expected_assessment_status
        ):
            raise ValueError("ReThink batch assessment status contradicts runtime assessment")
    return model.model_dump(mode="json")
