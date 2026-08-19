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
    profile_sha256: str | None = None
    schema_name: str | None = None
    context_sha256: str | None = None
    tool_query_ids: tuple[str, ...] = ()


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
    wild_type_sites: str
    protein_context_id: str
    visible_observations: list[dict[str, Any]]
    previous_hypothesis_id: str | None
    previous_hypothesis_assessment: dict[str, Any] | None
    knowledge_graph: dict[str, Any] | None = None
    kg_interaction: dict[str, Any] | None = None
    critic_revision: dict[str, Any] | None = None

    @field_validator("mutable_positions", mode="before")
    @classmethod
    def normalize_json_positions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ReThinkContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    run_id: str
    round_id: int = Field(ge=0)
    visible_baseline: float
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
