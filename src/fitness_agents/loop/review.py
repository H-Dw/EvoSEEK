from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from fitness_agents.agents.critic import CriticAgent
from fitness_agents.agents.output_guards import RevisionConstraints
from fitness_agents.contracts.agent_io import RoleActivationState
from fitness_agents.contracts.batch_review import BatchReviewContext, ControlFeasibilityReceipt
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


class HypothesisRevisionRequested(ReviewRejected):
    """Critic asked for a new hypothesis rather than only a new batch."""

    def __init__(self, message: str, *, decisions: Sequence[CritiqueDecision]) -> None:
        super().__init__(message, decisions=decisions)
        self.decision = decisions[-1]


class ControlFeasibilityError(ReviewRejected):
    """Requested controls cannot be assembled from the frozen design universe."""

    def __init__(
        self,
        receipt: ControlFeasibilityReceipt,
        *,
        decisions: Sequence[CritiqueDecision],
    ) -> None:
        super().__init__(
            f"Control feasibility gate failed: {receipt.reason}", decisions=decisions
        )
        self.receipt = receipt


@dataclass(frozen=True)
class ReviewLoopResult:
    draft: DraftBatch
    report: ConflictReport
    decision: CritiqueDecision
    approved_batch: ApprovedBatch
    attempts: tuple[CritiqueDecision, ...]


BATCH_REVISION_ACTIONS = frozenset(
    {
        RequiredChangeAction.EXCLUDE_CANDIDATE,
        RequiredChangeAction.REPLACE_CANDIDATE,
        RequiredChangeAction.MAKE_FALSIFICATION_EXECUTABLE,
        RequiredChangeAction.ADD_CONTROL,
        RequiredChangeAction.INCREASE_DIVERSITY,
        RequiredChangeAction.ADD_EXPLORATION_QUOTA,
        RequiredChangeAction.REDUCE_MUTATION_DEPTH,
    }
)
HYPOTHESIS_REVISION_ACTIONS = frozenset(
    {
        RequiredChangeAction.REGENERATE_WITH_CONSTRAINTS,
        RequiredChangeAction.REQUEST_EVIDENCE,
        RequiredChangeAction.ADD_COUNTEREVIDENCE_SEARCH,
        RequiredChangeAction.RELAX_SOFT_PRIOR,
    }
)


@dataclass(frozen=True)
class RevisionPlan:
    constraints: RevisionConstraints = field(default_factory=RevisionConstraints)
    exclusions: set[str] = field(default_factory=set)
    regenerate_hypothesis: bool = False
    executable: bool = False


class RevisionPlanner:
    """Applies only explicit, allow-listed changes; it never edits sequences directly."""

    def exclusions(self, decision: CritiqueDecision) -> set[str]:
        return self.plan(decision).exclusions

    def plan(self, decision: CritiqueDecision) -> RevisionPlan:
        excluded: set[str] = set()
        constraints = RevisionConstraints()
        actions = {change.action for change in decision.required_changes}
        for change in decision.required_changes:
            if change.action in {
                RequiredChangeAction.EXCLUDE_CANDIDATE,
                RequiredChangeAction.REPLACE_CANDIDATE,
            }:
                excluded.update(change.target_ids)
            elif change.action is RequiredChangeAction.ADD_CONTROL:
                requested = change.parameters.get("control_count")
                constraints = replace(
                    constraints,
                    require_controls=True,
                    required_control_count=(
                        int(requested) if requested is not None else 2
                    ),
                )
            elif change.action is RequiredChangeAction.INCREASE_DIVERSITY:
                requested = change.parameters.get("minimum_batch_distance")
                constraints = replace(
                    constraints,
                    increase_diversity=True,
                    minimum_batch_distance=(
                        int(requested) if requested is not None else None
                    ),
                )
            elif change.action is RequiredChangeAction.ADD_EXPLORATION_QUOTA:
                constraints = replace(constraints, add_exploration=True)
            elif change.action is RequiredChangeAction.REDUCE_MUTATION_DEPTH:
                constraints = replace(constraints, reduce_mutation_depth=True)
            elif change.action in HYPOTHESIS_REVISION_ACTIONS:
                constraints = replace(constraints, regenerate_hypothesis=True)
        regenerate = bool(actions.intersection(HYPOTHESIS_REVISION_ACTIONS))
        executable = bool(actions) and actions.issubset(
            BATCH_REVISION_ACTIONS.union(HYPOTHESIS_REVISION_ACTIONS)
        )
        if RequiredChangeAction.MAKE_FALSIFICATION_EXECUTABLE in actions:
            executable = True
        return RevisionPlan(
            constraints=constraints,
            exclusions=excluded,
            regenerate_hypothesis=regenerate,
            executable=executable
            or bool(excluded)
            or any(
                (
                    constraints.require_controls,
                    constraints.increase_diversity,
                    constraints.add_exploration,
                    constraints.reduce_mutation_depth,
                )
            ),
        )


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
        draft_builder: Callable[..., DraftBatch],
        variants: Mapping[str, Variant],
        predictions: Mapping[str, Prediction],
        evidence: Mapping[str, Sequence[Evidence]],
        revealed_ids: set[str],
        pending_ids: set[str],
        allowed_ids: set[str],
        expected_batch_size: int,
        context_evidence: Sequence[Evidence] = (),
        hypothesis: Any | None = None,
        activation_state: RoleActivationState | dict[str, Any] | None = None,
        review_context_provider: Callable[[DraftBatch], BatchReviewContext] | None = None,
        on_attempt: Callable[[DraftBatch, ConflictReport, CritiqueDecision], Any] | None = None,
        on_attempt_start: Callable[[DraftBatch, ConflictReport], Any] | None = None,
    ) -> ReviewLoopResult:
        attempts: list[CritiqueDecision] = []
        exclusions: set[str] = set()
        constraints = RevisionConstraints()
        parent_id: str | None = None
        for attempt in range(self.max_revision_attempts + 1):
            try:
                draft = draft_builder(attempt, parent_id, exclusions, constraints)
            except TypeError:
                draft = draft_builder(attempt, parent_id, exclusions)
            review_context = (
                review_context_provider(draft)
                if review_context_provider is not None
                else None
            )
            report = self.validator.validate(
                draft,
                variants=variants,
                predictions=predictions,
                evidence=evidence,
                revealed_ids=revealed_ids,
                pending_ids=pending_ids,
                allowed_ids=allowed_ids,
                expected_batch_size=expected_batch_size,
                prediction_decision_eligible=(
                    {
                        variant_id: card.decision_eligible
                        for variant_id, card in review_context.prediction_status_by_id.items()
                    }
                    if review_context is not None
                    else None
                ),
            )
            if on_attempt_start is not None:
                on_attempt_start(draft, report)
            if (
                review_context is not None
                and review_context.review_controls
                and review_context.control_feasibility is not None
                and not review_context.control_feasibility.feasible
            ):
                raise ControlFeasibilityError(
                    review_context.control_feasibility, decisions=attempts
                )
            decision = self.critic.review(
                draft=draft,
                variants=variants,
                predictions=predictions,
                evidence=evidence,
                conflict_report=report,
                context_evidence=context_evidence,
                hypothesis=hypothesis,
                activation_state=activation_state,
                batch_review_context=review_context,
            )
            attempts.append(decision)
            if on_attempt is not None:
                on_attempt(draft, report, decision)
            if decision.verdict is ReviewVerdict.APPROVE:
                approved = self.gateway.approve(draft=draft, report=report, decision=decision)
                return ReviewLoopResult(draft, report, decision, approved, tuple(attempts))
            if decision.verdict is ReviewVerdict.REJECT:
                raise ReviewRejected(
                    decision.summary or "Critic rejected the draft", decisions=attempts
                )
            if attempt >= self.max_revision_attempts:
                raise ReviewRejected("Critic revision limit exhausted", decisions=attempts)
            plan = self.revision_planner.plan(decision)
            if plan.regenerate_hypothesis:
                raise HypothesisRevisionRequested(
                    "Critic requested a new hypothesis with revision constraints",
                    decisions=attempts,
                )
            if not plan.executable:
                raise ReviewRejected(
                    "Revision request cannot be executed safely", decisions=attempts
                )
            exclusions.update(plan.exclusions)
            constraints = constraints.merge(plan.constraints)
            parent_id = draft.draft_batch_id
        raise AssertionError("Bounded review loop terminated unexpectedly")
