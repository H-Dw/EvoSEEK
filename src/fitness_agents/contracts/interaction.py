"""Public contracts shared by CLI and local interaction adapters."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CANONICAL_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")


class UserConstraintIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    position_policy: Literal["all", "include", "all_except"] = "all"
    include_positions: tuple[int, ...] = ()
    exclude_positions: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_positions(self) -> UserConstraintIntent:
        if self.position_policy == "include" and not self.include_positions:
            raise ValueError("include position policy requires at least one position")
        if len(set(self.include_positions)) != len(self.include_positions):
            raise ValueError("include positions must be unique")
        if len(set(self.exclude_positions)) != len(self.exclude_positions):
            raise ValueError("exclude positions must be unique")
        if any(item < 0 for item in self.include_positions + self.exclude_positions):
            raise ValueError("positions must be non-negative task coordinates")
        return self


class EvolutionIntent(BaseModel):
    """Strict intent produced before any run directory or model call is created."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    action: Literal["design", "explain", "status", "cancel"] = "design"
    objective_text: str | None = None
    assay_description: str | None = None
    desired_direction: Literal["maximize", "minimize"] | None = "maximize"
    sequence_source: Literal["message", "attachment", "configured"] | None = None
    reference_sequence: str | None = None
    reference_id: str | None = None
    requested_depth: int | None = Field(default=1, ge=1)
    requested_rounds: int | None = Field(default=1, ge=1)
    requested_budget: int | None = Field(default=None, ge=1)
    constraints: UserConstraintIntent = Field(default_factory=UserConstraintIntent)
    missing_fields: tuple[str, ...] = ()
    confirmation_summary: str

    @field_validator("reference_sequence")
    @classmethod
    def validate_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        sequence = "".join(value.split()).upper()
        invalid = sorted(set(sequence).difference(CANONICAL_RESIDUES))
        if not sequence or invalid:
            raise ValueError(
                f"reference sequence must use canonical amino-acid letters; invalid={invalid}"
            )
        return sequence


class OpenDesignRequestPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    preview_id: str
    action: Literal["design"] = "design"
    objective_text: str
    reference_id: str
    reference_length: int = Field(ge=1)
    design_space: Literal["open_design"] = "open_design"
    position_policy: Literal["all", "include", "all_except"]
    resolved_positions: tuple[int, ...]
    resolved_position_count: int = Field(ge=0)
    mutation_depth: int = Field(ge=1)
    budget: int = Field(ge=1)
    generated_candidate_count: int = Field(ge=0)
    supports_full_sequence: bool
    supports_generated_sequences: bool
    initial_data_source: str
    ready_for_confirmation: bool
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    confirmation_summary: str


class EvolutionRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    preview_id: str
    status: Literal["completed", "failed"]
    run_id: str | None = None
    public_message: str
    summary: dict[str, object] = Field(default_factory=dict)
    artifact_paths: tuple[str, ...] = ()


class EvolutionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    sequence_number: int = Field(ge=1)
    event_type: str
    phase: str
    public_message: str
    metrics: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
