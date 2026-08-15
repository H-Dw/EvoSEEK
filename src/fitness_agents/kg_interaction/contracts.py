from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryIntent(str, Enum):
    CONTEXT = "context"
    EXPLAIN = "explain"
    COMPARE = "compare"
    SUPPORT = "support"
    COUNTEREVIDENCE = "counterevidence"
    HISTORY = "history"
    UNCERTAINTY = "uncertainty"


@dataclass(frozen=True)
class KGQueryStep:
    step_id: str
    operator: str
    intent: QueryIntent
    arguments: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.step_id.strip():
            raise ValueError("step_id must not be empty")
        if not self.operator.strip():
            raise ValueError("operator must not be empty")


@dataclass(frozen=True)
class KGQueryPlan:
    plan_id: str
    objective: str
    steps: tuple[KGQueryStep, ...]
    max_tool_calls: int = 2

    def __post_init__(self) -> None:
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("query plan step IDs must be unique")


@dataclass(frozen=True)
class KGQueryContext:
    run_id: str
    round_id: int
    allowed_variant_ids: frozenset[str] | None = None
    max_rows: int = 12
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        if self.max_rows < 1:
            raise ValueError("max_rows must be positive")


@dataclass(frozen=True)
class EvidencePack:
    query_id: str
    operator: str
    as_of_round: int
    facts: tuple[dict[str, Any], ...] = ()
    predictions: tuple[dict[str, Any], ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    supporting_paths: tuple[dict[str, Any], ...] = ()
    counterevidence: tuple[dict[str, Any], ...] = ()
    directional_signals: tuple[dict[str, Any], ...] = ()
    caveats: tuple[str, ...] = ()
    provenance: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fact_count(self) -> int:
        return len(self.facts) + len(self.predictions) + len(self.evidence)

    @property
    def has_counterevidence(self) -> bool:
        return bool(self.counterevidence)


@dataclass(frozen=True)
class InteractionResult:
    plan_id: str
    packs: tuple[EvidencePack, ...]
    executed_steps: tuple[str, ...]
    skipped_steps: tuple[tuple[str, str], ...]
    stop_reason: str


class ChangeOperation(str, Enum):
    ADD_HYPOTHESIS = "add_hypothesis"
    LINK_EVIDENCE = "link_evidence"
    CHANGE_HYPOTHESIS_STATUS = "change_hypothesis_status"
    ADD_CURATED_CLAIM = "add_curated_claim"
    MERGE_ALIAS = "merge_alias"


@dataclass(frozen=True)
class KGChangeProposal:
    proposal_id: str
    actor: str
    operation: ChangeOperation
    target_id: str | None
    payload: dict[str, Any]
    evidence_ids: tuple[str, ...]
    idempotency_key: str
    confidence: float = 0.0


@dataclass(frozen=True)
class KGUpdateResult:
    proposal_id: str
    status: str
    transaction_id: str | None = None
    errors: tuple[str, ...] = ()
    created_ids: tuple[str, ...] = ()
