from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from inspect import signature
from typing import Any

from fitness_agents.agents.critic import CriticAgent
from fitness_agents.agents.output_guards import (
    CANONICAL_RESIDUES,
    DEFAULT_RESIDUE_CONSTRAINT_ARMS,
    REVISION_ARMS,
    ResidueSubstitutionConstraint,
    RevisionConstraints,
)
from fitness_agents.contracts.agent_io import RoleActivationState
from fitness_agents.contracts.batch_review import (
    BatchReviewContext,
    BatchRevisionFeedbackReceipt,
    ControlFeasibilityReceipt,
    ResidueSubstitutionCard,
    RevisionQuotaShortfallReceipt,
)
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


class RevisionConstraintInfeasible(ReviewRejected):
    """A revised pool cannot satisfy deterministic residue/quota constraints."""

    def __init__(
        self,
        receipt: RevisionQuotaShortfallReceipt,
        *,
        decisions: Sequence[CritiqueDecision] = (),
    ) -> None:
        super().__init__(receipt.code, decisions=decisions)
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

    @classmethod
    def _parse_excluded_substitution(
        cls,
        value: ResidueSubstitutionCard | dict[str, Any],
        *,
        allowed_positions: set[int] | None,
        wild_type_by_position: Mapping[int, str] | None,
    ) -> ResidueSubstitutionConstraint:
        card = ResidueSubstitutionCard.model_validate(value)
        position = card.position
        if allowed_positions is not None and position not in allowed_positions:
            raise ValueError(
                f"excluded residue position {position} is outside the design space"
            )
        from_residue = card.from_residue
        if (
            from_residue is not None
            and wild_type_by_position is not None
            and wild_type_by_position.get(position) != from_residue
        ):
            raise ValueError(
                f"excluded residue source {from_residue}{position} does not match wild type"
            )
        return ResidueSubstitutionConstraint(
            position=position,
            from_residue=from_residue,
            to_residue=card.to_residue,
        )

    def plan(
        self,
        decision: CritiqueDecision,
        *,
        allowed_positions: set[int] | None = None,
        wild_type_by_position: Mapping[int, str] | None = None,
    ) -> RevisionPlan:
        excluded: set[str] = set()
        constraints = RevisionConstraints()
        excluded_substitutions: set[ResidueSubstitutionConstraint] = set()
        required_residues: dict[int, set[str]] = {}
        applies_to_arms: set[str] = set()
        actions = {change.action for change in decision.required_changes}
        for change in decision.required_changes:
            for raw in change.parameters.get("excluded_substitutions", ()):
                excluded_substitutions.add(
                    self._parse_excluded_substitution(
                        raw,
                        allowed_positions=allowed_positions,
                        wild_type_by_position=wild_type_by_position,
                    )
                )
            for raw_position, raw_residues in dict(
                change.parameters.get("required_residues_by_position", {})
            ).items():
                position = int(raw_position)
                if allowed_positions is not None and position not in allowed_positions:
                    raise ValueError(
                        f"required residue position {position} is outside the design space"
                    )
                residues = {str(item).upper() for item in raw_residues}
                if not residues or residues.difference(CANONICAL_RESIDUES):
                    raise ValueError(
                        "required_residues_by_position must contain canonical residues"
                    )
                if position in required_residues:
                    required_residues[position].intersection_update(residues)
                else:
                    required_residues[position] = residues
            raw_arms = {
                str(item) for item in change.parameters.get("applies_to_arms", ())
            }
            if unknown_arms := raw_arms.difference(REVISION_ARMS):
                raise ValueError(f"unknown revision arms: {sorted(unknown_arms)}")
            applies_to_arms.update(raw_arms)
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
        if excluded_substitutions or required_residues:
            if not applies_to_arms:
                applies_to_arms.update(DEFAULT_RESIDUE_CONSTRAINT_ARMS)
            constraints = replace(
                constraints,
                excluded_substitutions=tuple(sorted(excluded_substitutions)),
                required_residues_by_position={
                    position: tuple(sorted(residues))
                    for position, residues in sorted(required_residues.items())
                },
                applies_to_arms=tuple(sorted(applies_to_arms)),
            )
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
                    constraints.has_residue_constraints,
                )
            ),
        )

    @staticmethod
    def feedback_receipt(
        decision: CritiqueDecision,
        plan: RevisionPlan,
    ) -> BatchRevisionFeedbackReceipt:
        issue_codes = {
            str(getattr(item.code, "value", item.code))
            for item in (*decision.candidate_issues, *decision.batch_level_risks)
        }
        return BatchRevisionFeedbackReceipt(
            previous_decision_id=decision.decision_id,
            previous_review_attempt=decision.review_attempt,
            issue_codes=tuple(sorted(issue_codes)),
            required_actions=tuple(
                sorted({item.action.value for item in decision.required_changes})
            ),
            excluded_candidate_ids=tuple(sorted(plan.exclusions)),
            excluded_substitutions=tuple(
                ResidueSubstitutionCard(
                    position=item.position,
                    from_residue=item.from_residue,
                    to_residue=item.to_residue,
                )
                for item in plan.constraints.excluded_substitutions
            ),
            required_residues_by_position={
                str(position): tuple(residues)
                for position, residues in sorted(
                    plan.constraints.required_residues_by_position.items()
                )
            },
            applies_to_arms=plan.constraints.applies_to_arms,
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
        position_to_index: Mapping[int, int] | None = None,
        review_context_provider: Callable[[DraftBatch], BatchReviewContext] | None = None,
        on_attempt: Callable[[DraftBatch, ConflictReport, CritiqueDecision], Any] | None = None,
        on_attempt_start: Callable[[DraftBatch, ConflictReport], Any] | None = None,
    ) -> ReviewLoopResult:
        attempts: list[CritiqueDecision] = []
        exclusions: set[str] = set()
        constraints = RevisionConstraints()
        parent_id: str | None = None
        revision_feedback: BatchRevisionFeedbackReceipt | None = None
        for attempt in range(self.max_revision_attempts + 1):
            try:
                parameters = signature(draft_builder).parameters
                builder_kwargs: dict[str, Any] = {}
                if "constraints" in parameters:
                    builder_kwargs["constraints"] = constraints
                if "revision_feedback" in parameters:
                    builder_kwargs["revision_feedback"] = revision_feedback
                draft = draft_builder(
                    attempt,
                    parent_id,
                    exclusions,
                    **builder_kwargs,
                )
            except RevisionConstraintInfeasible as error:
                raise RevisionConstraintInfeasible(
                    error.receipt, decisions=attempts
                ) from error
            review_context = (
                review_context_provider(draft)
                if review_context_provider is not None
                else None
            )
            if attempt > 0 and constraints.has_residue_constraints:
                validator_task = getattr(self.validator, "task", None)
                design_space = getattr(self.validator, "design_space", None)
                resolved_position_to_index = dict(
                    position_to_index
                    or (
                        design_space.position_to_sequence_index
                        if design_space is not None
                        else {
                            position: index
                            for index, position in enumerate(
                                validator_task.mutable_positions
                            )
                        }
                    )
                )
                if design_space is not None:
                    wild_type_by_position = {
                        position: design_space.reference_sequence[index]
                        for position, index in (
                            design_space.position_to_sequence_index.items()
                        )
                    }
                else:
                    wild_type_by_position = {
                        position: validator_task.wild_type_sites[index]
                        for index, position in enumerate(
                            validator_task.mutable_positions
                        )
                    }
                postcondition_failures: list[str] = []
                for candidate_id in draft.candidate_ids:
                    intent = (
                        review_context.candidate_intent_by_id.get(candidate_id)
                        if review_context is not None
                        else None
                    )
                    arm = intent.arm if intent is not None else "fallback"
                    if constraints.variant_violations(
                        variants[candidate_id],
                        arm=arm,
                        position_to_index=resolved_position_to_index,
                        wild_type_by_position=wild_type_by_position,
                    ):
                        postcondition_failures.append(candidate_id)
                if postcondition_failures:
                    raise RevisionConstraintInfeasible(
                        RevisionQuotaShortfallReceipt(
                            required_batch_size=expected_batch_size,
                            eligible_before_filter=len(variants),
                            eligible_after_filter=max(
                                0, len(draft.candidate_ids) - len(postcondition_failures)
                            ),
                            selected_count=len(draft.candidate_ids),
                            shortfall=max(
                                0,
                                expected_batch_size
                                - (
                                    len(draft.candidate_ids)
                                    - len(postcondition_failures)
                                ),
                            ),
                            quota_shortfalls={},
                            excluded_candidate_count=len(exclusions),
                            constraints_id=f"RC{draft.round_id:02d}-{attempt:02d}",
                            postcondition_failure_ids=tuple(
                                sorted(postcondition_failures)
                            ),
                        ),
                        decisions=attempts,
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
                hypothesis=hypothesis,
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
            try:
                validator_task = getattr(self.validator, "task", None)
                design_space = getattr(self.validator, "design_space", None)
                if validator_task is not None:
                    allowed_positions = set(validator_task.mutable_positions)
                    planner_wild_type = {
                        position: validator_task.wild_type_sites[index]
                        for index, position in enumerate(
                            validator_task.mutable_positions
                        )
                    }
                else:
                    allowed_positions = set(
                        design_space.allowed_mutation_positions
                    )
                    planner_wild_type = {
                        position: design_space.reference_sequence[index]
                        for position, index in (
                            design_space.position_to_sequence_index.items()
                        )
                    }
                plan = self.revision_planner.plan(
                    decision,
                    allowed_positions=allowed_positions,
                    wild_type_by_position=planner_wild_type,
                )
            except (TypeError, ValueError) as error:
                raise ReviewRejected(
                    f"Revision request cannot be executed safely: {error}",
                    decisions=attempts,
                ) from error
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
            revision_feedback = self.revision_planner.feedback_receipt(
                decision,
                RevisionPlan(
                    constraints=constraints,
                    exclusions=set(exclusions),
                    regenerate_hypothesis=False,
                    executable=True,
                ),
            )
            parent_id = draft.draft_batch_id
        raise AssertionError("Bounded review loop terminated unexpectedly")
