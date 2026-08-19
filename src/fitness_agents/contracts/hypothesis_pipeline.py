"""Typed contracts for the isolated, hierarchical hypothesis review graph."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ChannelName = Literal["physchem", "conservation", "structure"]
ReviewVerdictName = Literal["APPROVE", "REVISE", "REJECT"]

ISSUE_CODES = frozenset(
    {
        "CHANNEL_LEAKAGE",
        "CITATION_UNKNOWN",
        "CITATION_MISSING",
        "UNSUPPORTED_CLAIM",
        "COUNTEREVIDENCE_IGNORED",
        "OVERCONFIDENT",
        "UNTESTABLE",
        "FORMAT_INVALID",
        "EXPLANATION_MISSING",
        "CROSS_CHANNEL_CONFLICT",
    }
)
REQUIRED_ACTIONS = frozenset(
    {
        "REMOVE_FOREIGN_CONTEXT",
        "FIX_CITATIONS",
        "NARROW_CLAIM",
        "ADD_COUNTEREVIDENCE",
        "LOWER_CONFIDENCE",
        "MAKE_FALSIFIABLE",
        "FIX_FORMAT",
        "ADD_EXPLANATION",
        "RESOLVE_CHANNEL_CONFLICT",
    }
)


class ChannelEvidenceInput(BaseModel):
    """The complete and exclusive prompt payload for one child Scientist."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    run_id: str
    round_id: int = Field(ge=0)
    channel: ChannelName
    task: str
    mutable_positions: tuple[int, ...]
    wild_type_sites: str
    visible_observations: tuple[dict[str, Any], ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    kg_packs: tuple[dict[str, Any], ...] = ()
    retry_control: dict[str, Any] | None = None

    @field_validator("mutable_positions", "visible_observations", "evidence", "kg_packs", mode="before")
    @classmethod
    def normalize_tuples(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def enforce_channel_isolation(self) -> ChannelEvidenceInput:
        if len(self.mutable_positions) != len(self.wild_type_sites):
            raise ValueError("wild_type_sites must align with mutable_positions")
        foreign = {
            str(item.get("channel"))
            for item in self.evidence
            if item.get("channel") not in {None, self.channel}
        }
        for pack in self.kg_packs:
            for item in pack.get("evidence", ()):
                if isinstance(item, dict) and item.get("channel") not in {None, self.channel}:
                    foreign.add(str(item.get("channel")))
        if foreign:
            raise ValueError(
                f"{self.channel} child context contains foreign channels: {sorted(foreign)}"
            )
        return self

    @property
    def visible_evidence_ids(self) -> frozenset[str]:
        ids = {
            str(item["evidence_id"])
            for item in self.evidence
            if item.get("evidence_id")
        }
        ids.update(
            str(item["evidence_id"])
            for pack in self.kg_packs
            for item in pack.get("evidence", ())
            if isinstance(item, dict) and item.get("evidence_id")
        )
        return frozenset(ids)


class ChannelHypothesisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sub_hypothesis_id: str = Field(min_length=1, max_length=160)
    channel: ChannelName
    claim: str = Field(min_length=1, max_length=400)
    proposed_residues: dict[str, list[str]]
    evidence_ids: list[str] = Field(max_length=12)
    expected_effect: str = Field(min_length=1, max_length=400)
    counterevidence: list[str] = Field(max_length=8)
    uncertainty: str = Field(min_length=1, max_length=400)
    falsification_criterion: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def validate_residues(self) -> ChannelHypothesisOutput:
        canonical = set("ACDEFGHIKLMNPQRSTVWY")
        if not self.proposed_residues:
            raise ValueError("proposed_residues must not be empty")
        for position, residues in self.proposed_residues.items():
            int(position)
            if not residues or any(item not in canonical for item in residues):
                raise ValueError("proposed_residues must use canonical one-letter residues")
        return self


class HypothesisReviewIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    code: str
    severity: Literal["warning", "error", "blocker"]
    message: str = Field(min_length=1, max_length=400)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("code")
    @classmethod
    def allow_issue_code(cls, value: str) -> str:
        if value not in ISSUE_CODES:
            raise ValueError(f"unsupported issue code: {value}")
        return value


class HypothesisReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    decision_id: str = Field(min_length=1, max_length=160)
    verdict: ReviewVerdictName
    issues: list[HypothesisReviewIssue] = Field(max_length=12)
    required_changes: list[str] = Field(max_length=12)
    cited_evidence_ids: list[str] = Field(max_length=16)
    summary: str = Field(min_length=1, max_length=400)

    @field_validator("required_changes")
    @classmethod
    def allow_required_changes(cls, values: list[str]) -> list[str]:
        unknown = set(values).difference(REQUIRED_ACTIONS)
        if unknown:
            raise ValueError(f"unsupported required changes: {sorted(unknown)}")
        return values

    @model_validator(mode="after")
    def consistent_verdict(self) -> HypothesisReviewOutput:
        blockers = [item for item in self.issues if item.severity == "blocker"]
        if self.verdict == "APPROVE" and (blockers or self.required_changes):
            raise ValueError("APPROVE cannot contain blockers or required changes")
        if self.verdict == "REVISE" and not self.required_changes:
            raise ValueError("REVISE requires at least one allow-listed change")
        return self


class ApprovedSubHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    channel: ChannelName
    hypothesis: ChannelHypothesisOutput
    review: HypothesisReviewOutput
    attempt: int = Field(ge=0)
    input_sha256: str
    output_sha256: str

    @model_validator(mode="after")
    def must_be_approved(self) -> ApprovedSubHypothesis:
        if self.hypothesis.channel != self.channel or self.review.verdict != "APPROVE":
            raise ValueError("approved sub-hypothesis must match channel and APPROVE verdict")
        return self


class CrossChannelConflict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    position: int
    channels: tuple[ChannelName, ...]
    residue_sets: dict[str, tuple[str, ...]]
    resolution: Literal["main_scientist_must_resolve"] = "main_scientist_must_resolve"


class BranchReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    channel: ChannelName
    status: Literal["SUCCEEDED", "FAILED", "SKIPPED_UNAVAILABLE"]
    attempts: int = Field(ge=0)
    error_code: str | None = None
    input_chars: int | None = Field(default=None, ge=0)
    failure_category: str | None = None
    request_started: bool = False
    approved: ApprovedSubHypothesis | None = None


class HypothesisPipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    status: Literal["SUCCEEDED", "FAILED"]
    branches: tuple[BranchReceipt, ...]
    conflicts: tuple[CrossChannelConflict, ...]
    main_hypothesis: dict[str, Any] | None = None
    main_review: HypothesisReviewOutput | None = None
    main_attempts: int = Field(default=0, ge=0)
    degraded: bool = False
    failure_code: str | None = None


class CompletionManifest(BaseModel):
    """Run closure is separate from scientific/evaluation success."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    artifact_finalized: bool
    run_status: Literal["completed", "failed", "partial"]
    experiment_status: Literal["completed", "failed", "partial"]
    evaluation_status: Literal["not_evaluated", "eligible", "passed", "failed"]
    pass_eligible: bool
    expected_rounds: int = Field(ge=0)
    completed_rounds: int = Field(ge=0)
    aborted_rounds: int = Field(ge=0)
    required_node_failures: tuple[str, ...] = ()
    fallback_nodes: tuple[str, ...] = ()

    @field_validator("required_node_failures", "fallback_nodes", mode="before")
    @classmethod
    def normalize_json_tuples(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def prevent_false_pass(self) -> CompletionManifest:
        complete = (
            self.artifact_finalized
            and self.run_status == "completed"
            and self.experiment_status == "completed"
            and self.completed_rounds == self.expected_rounds
            and self.aborted_rounds == 0
            and not self.required_node_failures
            and not self.fallback_nodes
        )
        if self.pass_eligible != complete:
            raise ValueError("pass_eligible does not match completion gate")
        if self.evaluation_status == "passed" and not self.pass_eligible:
            raise ValueError("an incomplete or degraded experiment cannot be passed")
        return self
