"""Validated role inputs and observability-only trace metadata."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentTraceContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    run_id: str
    round_id: int = Field(ge=0)
    role: Literal["scientist", "rethink", "critic"]
    variant_id: str | None = None
    request_id: str | None = None
    profile: str | None = None
    schema_name: str | None = None
    tool_query_ids: tuple[str, ...] = ()


class RoleActivationState(BaseModel):
    """Observed execution route supplied to a role; never an authority grant."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    role: Literal["scientist", "critic", "rethink"] = "scientist"
    design_space: Literal["closed_pool", "open_design"] = "closed_pool"
    candidate_source: Literal["candidate_pool", "generated_from_reference"] = "candidate_pool"
    candidate_pool_consulted: bool = True
    position_policy: str = "configured"
    selection_driver: str = "agent_uq"
    active_learning_enabled: bool = False
    fitness_predictors_used_for_generation: bool = False
    rag_configured: bool = False
    rag_context_visible: bool = False
    rag_retrieval_performed: bool = False
    rag_evidence_present: bool = False
    kg_configured: bool = False
    kg_interaction_enabled: bool = False
    configured_kg_tools: tuple[str, ...] = ()
    executed_kg_tools: tuple[str, ...] = ()
    kg_tool_results_present: bool = False
    available_evidence_channels: tuple[str, ...] = ()
    unavailable_evidence_channels: tuple[str, ...] = ()

    @field_validator(
        "configured_kg_tools",
        "executed_kg_tools",
        "available_evidence_channels",
        "unavailable_evidence_channels",
        mode="before",
    )
    @classmethod
    def normalize_json_tuples(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_route_consistency(self) -> RoleActivationState:
        if self.design_space == "closed_pool":
            if self.candidate_source != "candidate_pool" or not self.candidate_pool_consulted:
                raise ValueError("closed_pool requires a consulted candidate_pool")
        elif self.candidate_source != "generated_from_reference" or self.candidate_pool_consulted:
            raise ValueError("open_design must generate from the reference without a pool")
        if self.rag_evidence_present and not self.rag_context_visible:
            raise ValueError("RAG evidence cannot be present when RAG context is hidden")
        if self.kg_tool_results_present and not self.kg_configured:
            raise ValueError("KG tool results require a configured KG")
        return self


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

    @field_validator("approved_subhypotheses", "cross_channel_conflicts", mode="before")
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
    decisive_criterion_ids: tuple[str, ...] = ()
    unresolved_criterion_ids: tuple[str, ...] = ()
    evaluator_version: str = Field(min_length=1)

    @field_validator("decisive_criterion_ids", "unresolved_criterion_ids", mode="before")
    @classmethod
    def normalize_criterion_ids(cls, value: Any) -> Any:
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


class ReThinkCandidateCard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    variant_id: str = Field(min_length=1)
    mutation_notation: str = Field(min_length=1)
    agent_reason: str
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


class ReThinkContextInput(BaseModel):
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
    candidates: list[ReThinkCandidateCard]

    @model_validator(mode="after")
    def unique_candidates(self) -> ReThinkContextInput:
        ids = [item.variant_id for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("ReThink candidates must have unique variant_id values")
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
        return frozenset(item.variant_id for item in self.candidates)
