"""Strict, auditable contracts for the two-stage Researcher planner.

The visible outputs are plans and receipts only.  They intentionally contain no
free-form chain of thought and grant no tool-execution authority to the model.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FacetName = Literal[
    "record_type",
    "knowledge_type",
    "question_leaf_id",
    "decision_slot",
    "task_route",
    "feature_channel",
    "required_input",
    "permission",
    "expected_direction",
    "stage",
    "evidence_role",
]
RetrievalIntent = Literal["support", "counterevidence", "boundary"]
FeatureChannel = Literal["physchem", "conservation", "structure"]
FeatureFocus = Literal[
    "site_deltas",
    "global_sequence_deltas",
    "special_flags",
    "site_log_odds",
    "pairwise_signal",
    "profile_quality",
    "solvent_exposure",
    "contact_geometry",
    "interface_contacts",
    "backbone_geometry",
    "interaction_flags",
]

FEATURE_FOCUS_BY_CHANNEL: dict[str, frozenset[str]] = {
    "physchem": frozenset(
        {"site_deltas", "global_sequence_deltas", "special_flags"}
    ),
    "conservation": frozenset(
        {"site_log_odds", "pairwise_signal", "profile_quality"}
    ),
    "structure": frozenset(
        {
            "solvent_exposure",
            "contact_geometry",
            "interface_contacts",
            "backbone_geometry",
            "interaction_flags",
        }
    ),
}


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ResearcherAssayContext(_StrictFrozenModel):
    assay_id: str = Field(min_length=1, max_length=160)
    objective: str = Field(min_length=1, max_length=1000)
    fitness_scale: str = Field(min_length=1, max_length=160)
    optimization_direction: Literal["higher_is_better", "lower_is_better"]
    conditions: dict[str, Any] = Field(default_factory=dict)


class ResearcherSampleCard(_StrictFrozenModel):
    sample_id: str = Field(pattern=r"^S[1-9]\d*$")
    observation_id: str = Field(min_length=1, max_length=200)
    measured_fitness: float
    round_revealed: int = Field(ge=0)
    source: str = Field(min_length=1, max_length=200)
    mutated_positions: tuple[int, ...] = ()

    @field_validator("mutated_positions", mode="before")
    @classmethod
    def normalize_positions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ResearcherFacetCatalog(_StrictFrozenModel):
    allowed_values: dict[FacetName, tuple[str, ...]] = Field(default_factory=dict)

    @field_validator("allowed_values", mode="before")
    @classmethod
    def normalize_values(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {
            str(key): tuple(str(item) for item in values)
            for key, values in value.items()
        }


class ResearcherToolCard(_StrictFrozenModel):
    tool_id: str = Field(min_length=1, max_length=100)
    channel: FeatureChannel
    allowed_positions: tuple[int, ...] = Field(min_length=1)
    allowed_focus: tuple[FeatureFocus, ...]
    projection_only: Literal[True] = True

    @field_validator("allowed_positions", "allowed_focus", mode="before")
    @classmethod
    def normalize_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def focus_matches_channel(self) -> ResearcherToolCard:
        if not self.allowed_focus or not set(self.allowed_focus).issubset(
            FEATURE_FOCUS_BY_CHANNEL[self.channel]
        ):
            raise ValueError("Tool focus must be a non-empty subset of its channel")
        if any(position < 1 for position in self.allowed_positions) or len(
            set(self.allowed_positions)
        ) != len(self.allowed_positions):
            raise ValueError("Tool positions must be unique positive integers")
        return self


class ResearcherKnowledgeRecordCard(_StrictFrozenModel):
    record_id: str = Field(min_length=1, max_length=240)
    record_type: Literal["atomic_claim", "logic_unit", "knowledge_decision_card"]
    retrieval_text: str = Field(min_length=1, max_length=5000)
    knowledge_type: str = Field(min_length=1, max_length=100)
    permission: str = Field(min_length=1, max_length=100)
    scientific_quality: dict[str, Any] = Field(default_factory=dict)
    task_applicability: dict[str, Any] = Field(default_factory=dict)
    boundary_conditions: tuple[str, ...] = ()
    counterclaims: tuple[str, ...] = ()
    abstain_if: tuple[str, ...] = ()
    facets: dict[FacetName, tuple[str, ...]] = Field(default_factory=dict)

    @field_validator("boundary_conditions", "counterclaims", "abstain_if", mode="before")
    @classmethod
    def normalize_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("facets", mode="before")
    @classmethod
    def normalize_facets(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {
            str(key): tuple(str(item) for item in values)
            for key, values in value.items()
        }


class ResearcherContextInput(_StrictFrozenModel):
    """Round-visible planning projection supplied to either Researcher phase."""

    schema_version: Literal["researcher-context:v1"] = "researcher-context:v1"
    phase: Literal["external_retrieval", "feature_evidence"]
    run_id: str = Field(min_length=1, max_length=240)
    round_id: int = Field(ge=0)
    task: str = Field(min_length=1, max_length=1000)
    assay: ResearcherAssayContext
    measurement_kg: tuple[ResearcherSampleCard, ...]
    prior_hypothesis_assessment: dict[str, Any] | None = None
    prior_hypothesis_reflection: dict[str, Any] | None = None
    sample_map: tuple[ResearcherSampleCard, ...] = ()
    rag_records: tuple[ResearcherKnowledgeRecordCard, ...] = ()
    facet_catalog: ResearcherFacetCatalog
    tool_catalog: tuple[ResearcherToolCard, ...] = ()

    @field_validator("measurement_kg", "sample_map", "rag_records", "tool_catalog", mode="before")
    @classmethod
    def normalize_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def phase_boundary(self) -> ResearcherContextInput:
        if self.phase == "external_retrieval" and (
            self.rag_records or self.sample_map or self.tool_catalog
        ):
            raise ValueError(
                "Phase A cannot receive RAG records, request-local samples, or feature tools"
            )
        if self.phase == "feature_evidence" and not self.tool_catalog:
            raise ValueError("Phase B requires an allow-listed feature tool catalog")
        return self


class RetrievalNeed(_StrictFrozenModel):
    need_id: str = Field(pattern=r"^N[1-9]\d*$")
    intent: RetrievalIntent
    scientific_question: str = Field(min_length=20, max_length=700)
    facets: dict[FacetName, tuple[str, ...]] = Field(default_factory=dict)
    top_k: int = Field(default=2, ge=1, le=2)
    rationale: str = Field(min_length=10, max_length=500)

    @field_validator("scientific_question")
    @classmethod
    def english_scientific_question(cls, value: str) -> str:
        if re.search(r"[\u3400-\u9fff]", value) or not re.search(r"[A-Za-z]", value):
            raise ValueError("scientific_question must be written in English")
        return value.strip()

    @field_validator("facets", mode="before")
    @classmethod
    def normalize_facets(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {
            str(key): tuple(str(item) for item in values)
            for key, values in value.items()
        }


class ExternalRetrievalPlan(_StrictFrozenModel):
    schema_version: Literal["external-retrieval-plan:v1"] = "external-retrieval-plan:v1"
    decision: Literal["PLAN", "ABSTAIN"]
    evidence_gap: str = Field(max_length=700)
    needs: tuple[RetrievalNeed, ...] = Field(default=(), max_length=3)
    abstention_reason: str | None = Field(default=None, max_length=500)

    @field_validator("needs", mode="before")
    @classmethod
    def normalize_needs(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def decision_is_coherent(self) -> ExternalRetrievalPlan:
        if self.decision == "ABSTAIN":
            if self.needs or not self.abstention_reason:
                raise ValueError("ABSTAIN requires a reason and no retrieval needs")
            return self
        if not self.evidence_gap.strip() or not self.needs:
            raise ValueError("PLAN requires an evidence gap and 1-3 retrieval needs")
        if any(item.intent == "support" for item in self.needs) and not any(
            item.intent in {"counterevidence", "boundary"} for item in self.needs
        ):
            raise ValueError(
                "A support retrieval must be paired with counterevidence or boundary retrieval"
            )
        if len({item.need_id for item in self.needs}) != len(self.needs):
            raise ValueError("Retrieval need IDs must be unique")
        return self


class FeatureEvidenceNeed(_StrictFrozenModel):
    request_id: str = Field(pattern=r"^F[1-9]\d*$")
    sample_id: str = Field(pattern=r"^S[1-9]\d*$")
    channel: FeatureChannel
    positions: tuple[int, ...] = Field(min_length=1)
    focus: tuple[FeatureFocus, ...] = Field(min_length=1)
    information_need: str = Field(min_length=10, max_length=700)
    rationale: str = Field(min_length=10, max_length=500)

    @field_validator("positions", "focus", mode="before")
    @classmethod
    def normalize_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def request_is_allowlisted(self) -> FeatureEvidenceNeed:
        if any(position < 1 for position in self.positions):
            raise ValueError("Feature positions must be positive")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("Feature positions must be unique")
        if not set(self.focus).issubset(FEATURE_FOCUS_BY_CHANNEL[self.channel]):
            raise ValueError("Feature focus does not belong to the requested channel")
        if len(set(self.focus)) != len(self.focus):
            raise ValueError("Feature focus values must be unique")
        return self


class FeatureEvidencePlan(_StrictFrozenModel):
    schema_version: Literal["feature-evidence-plan:v1"] = "feature-evidence-plan:v1"
    decision: Literal["PLAN", "ABSTAIN"]
    needs: tuple[FeatureEvidenceNeed, ...] = Field(default=(), max_length=6)
    abstention_reason: str | None = Field(default=None, max_length=500)

    @field_validator("needs", mode="before")
    @classmethod
    def normalize_needs(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def decision_is_coherent(self) -> FeatureEvidencePlan:
        if self.decision == "ABSTAIN":
            if self.needs or not self.abstention_reason:
                raise ValueError("ABSTAIN requires a reason and no feature needs")
            return self
        if not self.needs:
            raise ValueError("PLAN requires at least one feature request")
        if len({item.request_id for item in self.needs}) != len(self.needs):
            raise ValueError("Feature request IDs must be unique")
        return self


class ResearcherRoundReceipt(_StrictFrozenModel):
    schema_version: Literal["researcher-round-receipt:v1"] = (
        "researcher-round-receipt:v1"
    )
    run_id: str
    round_id: int = Field(ge=0)
    mode: Literal["two_stage"] = "two_stage"
    profile: str
    profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    kg_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_plan: ExternalRetrievalPlan | None = None
    feature_plan: FeatureEvidencePlan | None = None
    query_ids: tuple[str, ...] = ()
    record_ids: tuple[str, ...] = ()
    tool_query_ids: tuple[str, ...] = ()
    skipped: tuple[dict[str, str], ...] = ()
    rejected: tuple[dict[str, str], ...] = ()
    budget_used: dict[str, int] = Field(default_factory=dict)

    @field_validator(
        "query_ids", "record_ids", "tool_query_ids", "skipped", "rejected", mode="before"
    )
    @classmethod
    def normalize_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value
