"""Shared role observability contracts used by both ReThink modes."""

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
    candidate_source: Literal["candidate_pool", "generated_from_reference"] = (
        "candidate_pool"
    )
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
