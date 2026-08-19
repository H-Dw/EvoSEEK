from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from enum import Enum
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
    IssueScope,
    IssueSeverity,
    MutationConflict,
    Prediction,
    RequiredChangeAction,
    ReviewVerdict,
    Variant,
)
from fitness_agents.mutation.conflicts import ResidueConflictDetector, SequenceConflictDetector


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def content_hash(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    payload = {
        "round_id": round_id,
        "review_attempt": review_attempt,
        "candidate_ids": ordered_ids,
        "variants": [variants[item] for item in ordered_ids],
        "predictions": [predictions[item] for item in ordered_ids],
        "evidence": {item: tuple(evidence.get(item, ())) for item in ordered_ids},
        "hypothesis_id": hypothesis_id,
        "falsification_spec": falsification_spec,
        "design_rationales": rationales,
    }
    batch_hash = content_hash(payload)
    return DraftBatch(
        draft_batch_id=f"draft:r{round_id}:a{review_attempt}:{batch_hash[:12]}",
        parent_draft_batch_id=parent_draft_batch_id,
        round_id=round_id,
        review_attempt=review_attempt,
        candidate_ids=ordered_ids,
        hypothesis_ids=(hypothesis_id,) if hypothesis_id else (),
        prediction_snapshot_id=f"prediction:{content_hash(payload['predictions'])[:16]}",
        evidence_snapshot_id=f"evidence:{content_hash(payload['evidence'])[:16]}",
        acquisition_snapshot_id=f"acquisition:{content_hash(ordered_ids)[:16]}",
        design_rationales=rationales,
        falsification_spec=falsification_spec,
        batch_hash=batch_hash,
    )


def recompute_draft_hash(
    draft: DraftBatch,
    *,
    variants: Mapping[str, Variant],
    predictions: Mapping[str, Prediction],
    evidence: Mapping[str, Sequence[Evidence]],
) -> str:
    payload = {
        "round_id": draft.round_id,
        "review_attempt": draft.review_attempt,
        "candidate_ids": draft.candidate_ids,
        "variants": [variants[item] for item in draft.candidate_ids if item in variants],
        "predictions": [predictions[item] for item in draft.candidate_ids if item in predictions],
        "evidence": {item: tuple(evidence.get(item, ())) for item in draft.candidate_ids},
        "hypothesis_id": draft.hypothesis_ids[0] if draft.hypothesis_ids else None,
        "falsification_spec": draft.falsification_spec,
        "design_rationales": draft.design_rationales,
    }
    return content_hash(payload)


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
                    conflict_id=(
                        "conflict:evidence:"
                        f"{content_hash(sorted(missing_rationale_evidence))[:16]}"
                    ),
                    code="MISSING_RATIONALE_EVIDENCE",
                    scope=IssueScope.EVIDENCE,
                    severity=IssueSeverity.BLOCKER,
                    message="Design rationale cites evidence outside the frozen snapshot",
                    evidence_ids=tuple(sorted(missing_rationale_evidence)),
                    hard=True,
                    detector=f"evidence_reference:{self.version}",
                )
            )
        current_hash = recompute_draft_hash(
            draft, variants=variants, predictions=predictions, evidence=evidence
        )
        if current_hash != draft.batch_hash:
            conflicts.append(
                MutationConflict(
                    conflict_id=f"conflict:hash:{current_hash[:16]}",
                    code="DRAFT_HASH_MISMATCH",
                    scope=IssueScope.SYSTEM,
                    severity=IssueSeverity.BLOCKER,
                    message="Draft contents no longer match the reviewed batch hash",
                    candidate_ids=draft.candidate_ids,
                    hard=True,
                    detector=f"batch_hash:{self.version}",
                )
            )
        return ConflictReport(
            report_id=f"validation:{uuid.uuid4().hex}",
            round_id=draft.round_id,
            conflicts=tuple(conflicts),
            validator_version=self.version,
            input_hash=draft.batch_hash,
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
        referenced_conflict_ids = {
            conflict_id
            for issue in decision.candidate_issues
            for conflict_id in issue.conflict_ids
        }.union(item.conflict_id for item in decision.evidence_conflicts)
        if unknown := referenced_conflict_ids.difference(visible_conflict_ids):
            raise ValueError(f"CritiqueDecision references unknown conflicts: {sorted(unknown)}")
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
        self._issued_receipts: dict[str, str] = {}

    def approve(
        self,
        *,
        draft: DraftBatch,
        report: ConflictReport,
        decision: CritiqueDecision,
    ) -> ApprovedBatch:
        if report.input_hash != draft.batch_hash or report.hard_conflicts:
            raise PermissionError("Draft does not have a clean, matching hard-validation receipt")
        if decision.verdict is not ReviewVerdict.APPROVE:
            raise PermissionError("Only an APPROVE decision can create an approval receipt")
        receipt_payload = {
            "draft_batch_id": draft.draft_batch_id,
            "round_id": draft.round_id,
            "candidate_ids": draft.candidate_ids,
            "batch_hash": draft.batch_hash,
            "report_id": report.report_id,
            "decision_id": decision.decision_id,
            "policy": self.version,
        }
        approved = ApprovedBatch(
            draft_batch_id=draft.draft_batch_id,
            round_id=draft.round_id,
            candidate_ids=draft.candidate_ids,
            batch_hash=draft.batch_hash,
            hard_validation_report_id=report.report_id,
            critique_decision_id=decision.decision_id,
            approval_policy_version=self.version,
            approval_receipt_hash=content_hash(receipt_payload),
        )
        self._issued_receipts[approved.approval_receipt_hash] = approved.batch_hash
        return approved

    def verify(self, batch: ApprovedBatch) -> None:
        expected = content_hash(
            {
                "draft_batch_id": batch.draft_batch_id,
                "round_id": batch.round_id,
                "candidate_ids": batch.candidate_ids,
                "batch_hash": batch.batch_hash,
                "report_id": batch.hard_validation_report_id,
                "decision_id": batch.critique_decision_id,
                "policy": batch.approval_policy_version,
            }
        )
        if batch.approval_policy_version != self.version or expected != batch.approval_receipt_hash:
            raise PermissionError("Approval receipt is invalid or has been modified")
        if self._issued_receipts.get(batch.approval_receipt_hash) != batch.batch_hash:
            raise PermissionError("Approval receipt was not issued by this campaign gateway")
