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


class ReThinkContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    run_id: str
    round_id: int = Field(ge=0)
    visible_baseline: float
    activation_state: RoleActivationState = Field(
        default_factory=lambda: RoleActivationState(role="rethink")
    )
    approved_hypothesis: dict[str, Any] | None = None
    final_critic_decision: dict[str, Any] | None = None
    candidates: list[dict[str, Any]]

    @model_validator(mode="after")
    def unique_candidates(self) -> ReThinkContextInput:
        ids = [str(item.get("variant_id", "")) for item in self.candidates]
        if any(not item for item in ids):
            raise ValueError("Every ReThink candidate requires variant_id")
        if len(ids) != len(set(ids)):
            raise ValueError("ReThink candidates must have unique variant_id values")
        return self

    @property
    def expected_variant_ids(self) -> frozenset[str]:
        return frozenset(str(item["variant_id"]) for item in self.candidates)
