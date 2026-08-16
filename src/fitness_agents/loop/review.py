from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fitness_agents.agents.critic import CriticAgent
from fitness_agents.contracts.schemas import (
    ApprovedBatch,
    ConflictReport,
    CritiqueDecision,
    DraftBatch,
    Evidence,
    Prediction,
    RequiredChangeAction,
    ReviewVerdict,
    Variant,
)
from fitness_agents.validation.batch import ApprovalGateway, BatchHardValidator


class ReviewRejected(RuntimeError):
    def __init__(self, message: str, *, decisions: Sequence[CritiqueDecision]) -> None:
        super().__init__(message)
        self.decisions = tuple(decisions)


@dataclass(frozen=True)
class ReviewLoopResult:
    draft: DraftBatch
    report: ConflictReport
    decision: CritiqueDecision
    approved_batch: ApprovedBatch
    attempts: tuple[CritiqueDecision, ...]


class RevisionPlanner:
    """Applies only explicit, allow-listed changes; it never edits sequences directly."""

    def exclusions(self, decision: CritiqueDecision) -> set[str]:
        excluded: set[str] = set()
        for change in decision.required_changes:
            if change.action in {
                RequiredChangeAction.EXCLUDE_CANDIDATE,
                RequiredChangeAction.REPLACE_CANDIDATE,
            }:
                excluded.update(change.target_ids)
        return excluded


class BoundedReviewLoop:
    def __init__(
        self,
        *,
        validator: BatchHardValidator,
        critic: CriticAgent,
        max_revision_attempts: int,
        gateway: ApprovalGateway | None = None,
    ) -> None:
        self.validator = validator
        self.critic = critic
        self.max_revision_attempts = max_revision_attempts
        self.gateway = gateway or ApprovalGateway()
        self.revision_planner = RevisionPlanner()

    def run(
        self,
        *,
        draft_builder: Callable[[int, str | None, set[str]], DraftBatch],
        variants: Mapping[str, Variant],
        predictions: Mapping[str, Prediction],
        evidence: Mapping[str, Sequence[Evidence]],
        revealed_ids: set[str],
        pending_ids: set[str],
        allowed_ids: set[str],
        expected_batch_size: int,
        on_attempt: Callable[[DraftBatch, ConflictReport, CritiqueDecision], Any] | None = None,
        on_attempt_start: Callable[[DraftBatch, ConflictReport], Any] | None = None,
    ) -> ReviewLoopResult:
        attempts: list[CritiqueDecision] = []
        exclusions: set[str] = set()
        parent_id: str | None = None
        for attempt in range(self.max_revision_attempts + 1):
            draft = draft_builder(attempt, parent_id, exclusions)
            report = self.validator.validate(
                draft,
                variants=variants,
                predictions=predictions,
                evidence=evidence,
                revealed_ids=revealed_ids,
                pending_ids=pending_ids,
                allowed_ids=allowed_ids,
                expected_batch_size=expected_batch_size,
            )
            if on_attempt_start is not None:
                on_attempt_start(draft, report)
            decision = self.critic.review(
                draft=draft,
                variants=variants,
                predictions=predictions,
                evidence=evidence,
                conflict_report=report,
            )
            attempts.append(decision)
            if on_attempt is not None:
                on_attempt(draft, report, decision)
            if decision.verdict is ReviewVerdict.APPROVE:
                approved = self.gateway.approve(draft=draft, report=report, decision=decision)
                return ReviewLoopResult(draft, report, decision, approved, tuple(attempts))
            if decision.verdict is ReviewVerdict.REJECT:
                raise ReviewRejected(decision.summary or "Critic rejected the draft", decisions=attempts)
            if attempt >= self.max_revision_attempts:
                raise ReviewRejected("Critic revision limit exhausted", decisions=attempts)
            new_exclusions = self.revision_planner.exclusions(decision)
            if not new_exclusions and all(
                change.action is not RequiredChangeAction.MAKE_FALSIFICATION_EXECUTABLE
                for change in decision.required_changes
            ):
                raise ReviewRejected("Revision request cannot be executed safely", decisions=attempts)
            exclusions.update(new_exclusions)
            parent_id = draft.draft_batch_id
        raise AssertionError("Bounded review loop terminated unexpectedly")
