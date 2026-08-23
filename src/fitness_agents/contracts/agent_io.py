"""Validated role inputs and observability-only trace metadata."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .role_io_shared import AgentTraceContext, RoleActivationState  # noqa: F401


class ScientistContextInput(BaseModel):
    """Only the sanitized, round-visible context supplied by CampaignRunner."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    run_id: str
    mode: str
    round_id: int = Field(ge=0)
    expected_hypothesis_id: str
    task: str
    protein_id: str
    objective: str
    mutable_positions: tuple[int, ...]
    allowed_mutation_positions: tuple[int, ...] | None = None
    sequence_context_scope: Literal[
        "configured_mutable_sites", "full_reference_sequence"
    ] = "configured_mutable_sites"
    computation_position_count: int | None = Field(default=None, ge=1)
    wild_type_sites: str
    protein_context_id: str
    design_space: Literal["closed_pool", "open_design"] = "closed_pool"
    position_policy: str = "configured"
    preference_policy: Literal["all_positions", "sparse_subset"] = "all_positions"
    max_preferred_positions: int = Field(default=12, ge=1)
    activation_state: RoleActivationState = Field(default_factory=RoleActivationState)
    visible_observations: list[dict[str, Any]]
    previous_hypothesis_id: str | None
    previous_hypothesis_assessment: dict[str, Any] | None
    previous_hypothesis_reflection: dict[str, Any] | None = None
    prior_hypothesis_memory: tuple[dict[str, Any], ...] = ()
    knowledge_graph: dict[str, Any] | None = None
    kg_interaction: dict[str, Any] | None = None
    approved_subhypotheses: tuple[dict[str, Any], ...] = ()
    cross_channel_conflicts: tuple[dict[str, Any], ...] = ()
    critic_revision: dict[str, Any] | None = None

    @field_validator("mutable_positions", mode="before")
    @classmethod
    def normalize_json_positions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("allowed_mutation_positions", mode="before")
    @classmethod
    def normalize_allowed_positions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator(
        "approved_subhypotheses",
        "cross_channel_conflicts",
        "prior_hypothesis_memory",
        mode="before",
    )
    @classmethod
    def normalize_hierarchical_payloads(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def consistent_position_authority(self) -> ScientistContextInput:
        if self.allowed_mutation_positions is None:
            object.__setattr__(self, "allowed_mutation_positions", self.mutable_positions)
        if self.computation_position_count is None:
            object.__setattr__(
                self, "computation_position_count", len(self.mutable_positions)
            )
        if self.mutable_positions != self.allowed_mutation_positions:
            raise ValueError(
                "mutable_positions compatibility alias must equal allowed_mutation_positions"
            )
        if len(self.wild_type_sites) != len(self.allowed_mutation_positions):
            raise ValueError("wild_type_sites must align with allowed mutation positions")
        return self


class ReThinkHypothesisCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    hypothesis_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    expected_outcome: str = Field(min_length=1)
    falsification_criterion: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ReThinkCriticDecisionCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    decision_id: str = Field(min_length=1)
    verdict: Literal["APPROVE"]
    summary: str
    cited_evidence_ids: tuple[str, ...] = ()

    @field_validator("cited_evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ReThinkAssessmentCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    assessment_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    falsification_spec_id: str = Field(min_length=1)
    status: Literal["SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"]
    criterion_results: tuple[CriterionReceiptCard, ...] = ()
    observation_ids: tuple[str, ...] = ()
    decisive_criterion_ids: tuple[str, ...] = ()
    unresolved_criterion_ids: tuple[str, ...] = ()
    evaluator_version: str = Field(min_length=1)

    @field_validator(
        "criterion_results",
        "observation_ids",
        "decisive_criterion_ids",
        "unresolved_criterion_ids",
        mode="before",
    )
    @classmethod
    def normalize_criterion_ids(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class CriterionReceiptCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    criterion_id: str = Field(min_length=1)
    signal: Literal["SUPPORT", "CONTRADICT", "UNRESOLVED"]
    metric_value: float | None = None
    comparator_value: float | None = None
    effect_size: float | None = None
    observation_ids: tuple[str, ...] = ()
    qc_status: str = Field(min_length=1)
    detector_name: str = Field(min_length=1)
    detector_version: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)

    @field_validator("observation_ids", mode="before")
    @classmethod
    def normalize_observation_ids(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ReThinkMeasurementContractCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    assay_id: str = Field(min_length=1)
    fitness_scale: str = Field(min_length=1)
    optimization_direction: Literal["higher_is_better", "lower_is_better"]


class ReThinkBaselineReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    value: float
    statistic: Literal["pre_round_visible_median"]
    source: Literal["revealed_observations_before_current_round"]


class ReThinkCriterionCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    criterion_id: str = Field(min_length=1)
    detector_name: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    expected_direction: str = Field(min_length=1)
    target_variant_ids: tuple[str, ...]
    comparator_variant_ids: tuple[str, ...]
    min_observations: int = Field(ge=1)
    missing_data_policy: str = Field(min_length=1)
    primary: bool

    @field_validator("target_variant_ids", "comparator_variant_ids", mode="before")
    @classmethod
    def normalize_ids(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ReThinkFalsificationSpecCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    spec_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    reduction_policy: str = Field(min_length=1)
    criteria: tuple[ReThinkCriterionCard, ...] = Field(min_length=1)

    @field_validator("criteria", mode="before")
    @classmethod
    def normalize_criteria(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ReThinkDryValidationCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    value: float
    uncertainty: float = Field(ge=0)
    ood_score: float = Field(ge=0)
    model_version: str = Field(min_length=1)
    source_kind: Literal["dry_validation", "active_posterior", "real_model"]
    decision_eligible: bool
    calibration_status: Literal["unknown", "uncalibrated", "calibrated"]
    prediction_status: Literal["evaluated"]


class ReThinkObservationCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    variant_id: str = Field(min_length=1)
    mutation_notation: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    wet_value: float
    dry_validations: tuple[ReThinkDryValidationCard, ...] = ()
    intent_arm: Literal[
        "hypothesis_target",
        "evidence_prior",
        "coverage_exploration",
        "matched_control",
        "fallback",
    ]
    matched_to: str | None = None
    allow_hypothesis_mismatch: bool = False
    falsification_role: Literal["target", "comparator", "not_in_primary_criterion"]

    @field_validator("evidence_ids", "dry_validations", mode="before")
    @classmethod
    def normalize_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ArmSummaryCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    arm: Literal[
        "hypothesis_target",
        "evidence_prior",
        "coverage_exploration",
        "matched_control",
        "fallback",
    ]
    sample_count: int = Field(ge=1)
    variant_ids: tuple[str, ...] = Field(min_length=1)
    wet_mean: float
    wet_min: float
    wet_max: float
    favorable_count: int = Field(ge=0)
    dry_wet_disagreement_count: int = Field(ge=0)

    @field_validator("variant_ids", mode="before")
    @classmethod
    def normalize_variant_ids(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class DryWetDisagreementCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    variant_id: str = Field(min_length=1)
    wet_value: float
    dry_mean: float
    residual: float
    max_ood_score: float = Field(ge=0)
    model_versions: tuple[str, ...] = ()

    @field_validator("model_versions", mode="before")
    @classmethod
    def normalize_model_versions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class RoundEvidenceDigest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    observation_count: int = Field(ge=0)
    observations: tuple[ReThinkObservationCard, ...] = ()
    arm_summaries: tuple[ArmSummaryCard, ...] = ()
    dry_wet_disagreements: tuple[DryWetDisagreementCard, ...] = ()
    criterion_receipts: tuple[CriterionReceiptCard, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    @field_validator(
        "observations",
        "arm_summaries",
        "dry_wet_disagreements",
        "criterion_receipts",
        "evidence_ids",
        mode="before",
    )
    @classmethod
    def normalize_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_counts(self) -> RoundEvidenceDigest:
        if self.observation_count != len(self.observations):
            raise ValueError("RoundEvidenceDigest observation_count must match observations")
        return self


class HypothesisReflectionContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    run_id: str
    round_id: int = Field(ge=0)
    visible_baseline: float
    baseline_receipt: ReThinkBaselineReceipt
    measurement_contract: ReThinkMeasurementContractCard
    activation_state: RoleActivationState = Field(
        default_factory=lambda: RoleActivationState(role="rethink")
    )
    approved_hypothesis: ReThinkHypothesisCard | None = None
    final_critic_decision: ReThinkCriticDecisionCard
    hypothesis_assessment: ReThinkAssessmentCard | None = None
    falsification_spec: ReThinkFalsificationSpecCard | None = None
    round_evidence_digest: RoundEvidenceDigest

    @model_validator(mode="after")
    def consistent_hypothesis_scope(self) -> HypothesisReflectionContextInput:
        ids = [item.variant_id for item in self.round_evidence_digest.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("ReThink observations must have unique variant_id values")
        if len(
            {
                self.approved_hypothesis is None,
                self.hypothesis_assessment is None,
                self.falsification_spec is None,
            }
        ) != 1:
            raise ValueError(
                "ReThink hypothesis, falsification spec, and deterministic assessment must be supplied together"
            )
        if (
            self.approved_hypothesis is not None
            and self.hypothesis_assessment is not None
            and self.approved_hypothesis.hypothesis_id
            != self.hypothesis_assessment.hypothesis_id
        ):
            raise ValueError("ReThink assessment must match the approved hypothesis")
        if self.visible_baseline != self.baseline_receipt.value:
            raise ValueError("visible_baseline must equal the typed baseline receipt")
        if (
            self.falsification_spec is not None
            and self.approved_hypothesis is not None
            and self.falsification_spec.hypothesis_id
            != self.approved_hypothesis.hypothesis_id
        ):
            raise ValueError("ReThink falsification spec must match the approved hypothesis")
        return self

    @property
    def expected_variant_ids(self) -> frozenset[str]:
        return frozenset(
            item.variant_id for item in self.round_evidence_digest.observations
        )


# Additive compatibility exports for the original candidate-level ReThink API.
from .rethink_sample_io import ReThinkCandidateCard, ReThinkContextInput  # noqa: F401
