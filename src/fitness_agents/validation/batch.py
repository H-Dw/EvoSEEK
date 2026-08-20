from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fitness_agents.config import CriticConfig, TaskConfig
from fitness_agents.contracts.schemas import (
    ApprovedBatch,
    ConflictReport,
    CritiqueDecision,
    DesignRationale,
    DraftBatch,
    Evidence,
    FalsificationReadiness,
    FalsificationSpec,
    Hypothesis,
    IssueScope,
    IssueSeverity,
    MutationConflict,
    Prediction,
    RequiredChangeAction,
    ReviewVerdict,
    Variant,
)
from fitness_agents.mutation.conflicts import ResidueConflictDetector, SequenceConflictDetector


def build_draft_batch(
    *,
    round_id: int,
    review_attempt: int,
    candidate_ids: Sequence[str],
    variants: Mapping[str, Variant],
    predictions: Mapping[str, Prediction],
    evidence: Mapping[str, Sequence[Evidence]],
    hypothesis_id: str | None,
    falsification_spec: FalsificationSpec | None,
    parent_draft_batch_id: str | None = None,
    rationale_claims: Mapping[str, str] | None = None,
) -> DraftBatch:
    ordered_ids = tuple(candidate_ids)
    rationale_claims = rationale_claims or {}
    rationales = tuple(
        DesignRationale(
            candidate_id=item,
            hypothesis_id=hypothesis_id,
            claim=rationale_claims.get(
                item, "Candidate selected by the configured acquisition policy."
            ),
            evidence_ids=tuple(entry.evidence_id for entry in evidence.get(item, ())),
            intended_test=(
                falsification_spec.human_readable_description if falsification_spec else "Optimization control"
            ),
        )
        for item in ordered_ids
    )
    return DraftBatch(
        draft_batch_id=f"B{round_id:02d}-{review_attempt:02d}",
        parent_draft_batch_id=parent_draft_batch_id,
        round_id=round_id,
        review_attempt=review_attempt,
        candidate_ids=ordered_ids,
        hypothesis_ids=(hypothesis_id,) if hypothesis_id else (),
        prediction_snapshot_id=f"P{round_id:02d}-{review_attempt:02d}",
        evidence_snapshot_id=f"E{round_id:02d}-{review_attempt:02d}",
        acquisition_snapshot_id=f"A{round_id:02d}-{review_attempt:02d}",
        design_rationales=rationales,
        falsification_spec=falsification_spec,
    )


class BatchHardValidator:
    version = "1.0.0"

    def __init__(self, task: TaskConfig, critic: CriticConfig) -> None:
        self.task = task
        self.residue = ResidueConflictDetector()
        self.sequence = SequenceConflictDetector(
            ood_warning_threshold=critic.ood_warning_threshold,
            model_disagreement_threshold=critic.model_disagreement_threshold,
            min_batch_distance=(
                critic.min_batch_distance if critic.review_diversity else 0
            ),
        )

    def validate(
        self,
        draft: DraftBatch,
        *,
        variants: Mapping[str, Variant],
        predictions: Mapping[str, Prediction],
        evidence: Mapping[str, Sequence[Evidence]],
        revealed_ids: set[str],
        pending_ids: set[str],
        allowed_ids: set[str],
        expected_batch_size: int,
        prediction_decision_eligible: Mapping[str, bool] | None = None,
        hypothesis: Hypothesis | None = None,
    ) -> ConflictReport:
        selected = [variants[item] for item in draft.candidate_ids if item in variants]
        conflicts = self.residue.detect(
            selected,
            wild_type_sites=self.task.wild_type_sites,
            mutable_positions=self.task.mutable_positions,
        )
        conflicts.extend(
            self.sequence.detect(
                selected,
                predictions=predictions,
                evidence=evidence,
                revealed_ids=revealed_ids,
                pending_ids=pending_ids,
                allowed_ids=allowed_ids,
                expected_batch_size=expected_batch_size,
                prediction_decision_eligible=prediction_decision_eligible,
            )
        )
        if hypothesis is not None and hypothesis.hard_residue_constraints:
            position_to_index = {
                position: index
                for index, position in enumerate(self.task.mutable_positions)
            }
            for candidate in selected:
                violations = {
                    position: tuple(allowed)
                    for position, allowed in hypothesis.hard_residue_constraints.items()
                    if position not in position_to_index
                    or position_to_index[position] >= len(candidate.variant)
                    or candidate.variant[position_to_index[position]] not in allowed
                }
                if violations:
                    conflicts.append(
                        MutationConflict(
                            conflict_id=f"C-HARD-{candidate.variant_id}",
                            code="HARD_RESIDUE_CONSTRAINT_VIOLATION",
                            scope=IssueScope.RESIDUE,
                            severity=IssueSeverity.BLOCKER,
                            message=(
                                "Candidate violates explicit hard_residue_constraints; "
                                "preferred_residues are not part of this gate"
                            ),
                            candidate_ids=(candidate.variant_id,),
                            hard=True,
                            detector=f"hard_residue_constraints:{self.version}",
                        )
                    )
        visible_evidence_ids = {
            item.evidence_id
            for candidate_id in draft.candidate_ids
            for item in evidence.get(candidate_id, ())
        }
        missing_rationale_evidence = {
            evidence_id
            for rationale in draft.design_rationales
            for evidence_id in rationale.evidence_ids
            if evidence_id not in visible_evidence_ids
        }
        if missing_rationale_evidence:
            conflicts.append(
                MutationConflict(
                    conflict_id="C-MISSING-EVIDENCE",
                    code="MISSING_RATIONALE_EVIDENCE",
                    scope=IssueScope.EVIDENCE,
                    severity=IssueSeverity.BLOCKER,
                    message="Design rationale cites evidence outside the frozen snapshot",
                    evidence_ids=tuple(sorted(missing_rationale_evidence)),
                    hard=True,
                    detector=f"evidence_reference:{self.version}",
                )
            )
        return ConflictReport(
            report_id=f"V{draft.round_id:02d}-{draft.review_attempt:02d}",
            round_id=draft.round_id,
            conflicts=tuple(conflicts),
            validator_version=self.version,
            draft_batch_id=draft.draft_batch_id,
        )


class CritiqueDecisionValidator:
    version = "1.0.0"

    def validate(
        self,
        decision: CritiqueDecision,
        *,
        draft: DraftBatch,
        report: ConflictReport,
        visible_evidence_ids: set[str],
        hypothesis: Hypothesis | None = None,
        batch_review_context: Any | None = None,
    ) -> None:
        if (decision.draft_batch_id, decision.round_id, decision.review_attempt) != (
            draft.draft_batch_id,
            draft.round_id,
            draft.review_attempt,
        ):
            raise ValueError("CritiqueDecision does not match the reviewed draft")
        visible_candidates = set(draft.candidate_ids)
        candidate_actions = {
            RequiredChangeAction.EXCLUDE_CANDIDATE,
            RequiredChangeAction.REPLACE_CANDIDATE,
            RequiredChangeAction.REDUCE_MUTATION_DEPTH,
        }
        referenced_candidates = {
            issue.candidate_id for issue in decision.candidate_issues
        }.union(
            *(
                set(change.target_ids)
                for change in decision.required_changes
                if change.action in candidate_actions
            )
        )
        if unknown := referenced_candidates.difference(visible_candidates):
            raise ValueError(f"CritiqueDecision references unknown candidates: {sorted(unknown)}")
        cited = set(decision.cited_evidence_ids)
        cited.update(
            evidence_id
            for issue in decision.candidate_issues
            for evidence_id in issue.evidence_ids
        )
        cited.update(
            evidence_id
            for conflict in decision.evidence_conflicts
            for evidence_id in conflict.supporting_ids + conflict.opposing_ids
        )
        cited.update(
            evidence_id
            for change in decision.required_changes
            for evidence_id in change.evidence_ids
        )
        if unknown := cited.difference(visible_evidence_ids):
            raise ValueError(f"CritiqueDecision references invisible evidence: {sorted(unknown)}")
        visible_conflict_ids = {item.conflict_id for item in report.conflicts}
        hard_code = "HARD_RESIDUE_CONSTRAINT_VIOLATION"
        referenced_conflict_ids = {
            conflict_id
            for issue in decision.candidate_issues
            if getattr(issue.code, "value", str(issue.code)) != hard_code
            for conflict_id in issue.conflict_ids
        }
        if unknown := referenced_conflict_ids.difference(visible_conflict_ids):
            raise ValueError(f"CritiqueDecision references unknown conflicts: {sorted(unknown)}")

        explicit_hard = (
            dict(hypothesis.hard_residue_constraints)
            if hypothesis is not None
            else {}
        )
        hard_conflicts = {
            item.conflict_id: item
            for item in report.conflicts
            if item.hard and item.code == hard_code
        }
        hard_issues = [
            item
            for item in decision.candidate_issues
            if getattr(item.code, "value", str(item.code)) == hard_code
        ]
        hard_risks = [
            item
            for item in decision.batch_level_risks
            if getattr(item.code, "value", str(item.code)) == hard_code
        ]
        if not explicit_hard and (hard_issues or hard_risks):
            raise ValueError(
                "HARD_RESIDUE_CONSTRAINT_VIOLATION is forbidden when "
                "hard_residue_constraints is empty"
            )
        if hard_risks:
            raise ValueError(
                "HARD_RESIDUE_CONSTRAINT_VIOLATION must be candidate-scoped and cite a "
                "deterministic hard-conflict ID"
            )
        for issue in hard_issues:
            if not issue.conflict_ids:
                raise ValueError(
                    "HARD_RESIDUE_CONSTRAINT_VIOLATION must cite a deterministic "
                    "hard-conflict ID"
                )
            for conflict_id in issue.conflict_ids:
                conflict = hard_conflicts.get(conflict_id)
                if conflict is None or issue.candidate_id not in conflict.candidate_ids:
                    raise ValueError(
                        "HARD_RESIDUE_CONSTRAINT_VIOLATION cites no matching deterministic "
                        "hard-conflict ID"
                    )

        for change in decision.required_changes:
            required = dict(change.parameters.get("required_residues_by_position", {}))
            if required and not explicit_hard:
                raise ValueError(
                    "required_residues_by_position cannot be derived from "
                    "preferred_residues; explicit hard_residue_constraints are required"
                )
            for raw_position, residues in required.items():
                position = int(raw_position)
                allowed = set(explicit_hard.get(position, ()))
                if not allowed or not set(residues).issubset(allowed):
                    raise ValueError(
                        "required_residues_by_position must be a subset of explicit "
                        "hard_residue_constraints"
                    )

        if batch_review_context is not None:
            from fitness_agents.contracts.batch_review import BatchReviewContext

            review_context = BatchReviewContext.model_validate(batch_review_context)
            excluded_targets = {
                target
                for change in decision.required_changes
                if change.action
                in {
                    RequiredChangeAction.EXCLUDE_CANDIDATE,
                    RequiredChangeAction.REPLACE_CANDIDATE,
                }
                for target in change.target_ids
            }
            soft_mismatches = set(review_context.soft_prior_mismatch_ids)
            for target in excluded_targets:
                intent = review_context.candidate_intent_by_id.get(target)
                has_deterministic_hard_conflict = any(
                    target in conflict.candidate_ids for conflict in report.hard_conflicts
                )
                has_independent_candidate_issue = any(
                    issue.candidate_id == target
                    and getattr(issue.code, "value", str(issue.code)) != hard_code
                    for issue in decision.candidate_issues
                )
                has_independent_batch_risk = any(
                    target in risk.candidate_ids for risk in decision.batch_level_risks
                )
                if (
                    intent is not None
                    and intent.arm == "matched_control"
                    and intent.allow_hypothesis_mismatch
                    and not has_deterministic_hard_conflict
                ):
                    raise ValueError(
                        "matched controls may intentionally violate preferred_residues; "
                        "soft prior mismatch cannot trigger exclusion"
                    )
                if (
                    target in soft_mismatches
                    and not has_deterministic_hard_conflict
                    and not has_independent_candidate_issue
                    and not has_independent_batch_risk
                ):
                    raise ValueError(
                        "soft prior mismatch alone cannot trigger candidate exclusion or replacement"
                    )
        if decision.verdict is ReviewVerdict.APPROVE:
            if report.hard_conflicts:
                raise ValueError("APPROVE cannot override unresolved hard conflicts")
            if decision.required_changes:
                raise ValueError("APPROVE cannot include required changes")
            if decision.falsification_readiness is not FalsificationReadiness.READY:
                raise ValueError("APPROVE requires falsification readiness")
        elif decision.verdict is ReviewVerdict.REVISE and not decision.required_changes:
            raise ValueError("REVISE requires at least one machine-executable change")
        elif decision.verdict is ReviewVerdict.REJECT:
            has_blocker = any(
                item.severity.value == "blocker" for item in decision.candidate_issues
            ) or any(item.severity.value == "blocker" for item in decision.batch_level_risks)
            has_abort = any(
                item.action is RequiredChangeAction.ABORT_ROUND
                for item in decision.required_changes
            )
            if not has_blocker and not has_abort:
                raise ValueError("REJECT requires a blocker or ABORT_ROUND")


class ApprovalGateway:
    version = "1.0.0"

    def __init__(self) -> None:
        self._issued_receipts: dict[str, tuple[object, ...]] = {}

    @staticmethod
    def _receipt_fields(batch: ApprovedBatch) -> tuple[object, ...]:
        """Return the ordinary, readable fields bound to an approval ID."""

        return (
            batch.draft_batch_id,
            batch.round_id,
            batch.candidate_ids,
            batch.hard_validation_report_id,
            batch.critique_decision_id,
            batch.approval_policy_version,
        )

    def approve(
        self,
        *,
        draft: DraftBatch,
        report: ConflictReport,
        decision: CritiqueDecision,
    ) -> ApprovedBatch:
        if report.draft_batch_id != draft.draft_batch_id or report.hard_conflicts:
            raise PermissionError("Draft does not have a clean, matching hard-validation receipt")
        if decision.verdict is not ReviewVerdict.APPROVE:
            raise PermissionError("Only an APPROVE decision can create an approval receipt")
        approval_id = f"AP{draft.round_id:02d}-{draft.review_attempt:02d}"
        approved = ApprovedBatch(
            draft_batch_id=draft.draft_batch_id,
            round_id=draft.round_id,
            candidate_ids=draft.candidate_ids,
            hard_validation_report_id=report.report_id,
            critique_decision_id=decision.decision_id,
            approval_policy_version=self.version,
            approval_id=approval_id,
        )
        self._issued_receipts[approved.approval_id] = self._receipt_fields(approved)
        return approved

    def verify(self, batch: ApprovedBatch) -> None:
        if batch.approval_policy_version != self.version:
            raise PermissionError("Approval policy version is invalid")
        issued = self._issued_receipts.get(batch.approval_id)
        if issued is None:
            raise PermissionError("Approval receipt was not issued by this campaign gateway")
        if issued != self._receipt_fields(batch):
            raise PermissionError("Approved batch was modified after approval")
