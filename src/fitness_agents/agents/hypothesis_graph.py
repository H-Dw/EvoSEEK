"""Explicit fan-out/fan-in Scientist--Critic graph with bounded revisions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import replace
from typing import Any

from fitness_agents.contracts.agent_io import ScientistContextInput
from fitness_agents.contracts.hypothesis_pipeline import (
    ApprovedSubHypothesis,
    BranchReceipt,
    ChannelName,
    CrossChannelConflict,
    HypothesisPipelineResult,
)
from fitness_agents.contracts.schemas import Evidence, Hypothesis
from fitness_agents.kg_interaction.contracts import InteractionResult

from .context_projection import (
    FEATURE_CHANNELS,
    KGContextPartitioner,
    canonical_sha256,
)
from .remote_llm import completion_receipt_snapshot, reset_completion_receipt
from .subscientist import validate_channel_hypothesis

MainProposer = Callable[..., Hypothesis]


def _failure_fields(
    error: Exception,
    *,
    completion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    completion = completion or {}
    return {
        "error_code": str(
            getattr(error, "error_code", f"{type(error).__name__}: {str(error)[:240]}")
        ),
        "input_chars": getattr(error, "input_chars", completion.get("input_chars")),
        "failure_category": getattr(
            error, "failure_category", completion.get("failure_category") or "runtime"
        ),
        "request_started": bool(
            getattr(error, "request_started", completion.get("request_started", False))
        ),
    }


def _conflicts(
    approved: tuple[ApprovedSubHypothesis, ...]
) -> tuple[CrossChannelConflict, ...]:
    by_position: dict[int, dict[ChannelName, tuple[str, ...]]] = defaultdict(dict)
    for item in approved:
        for raw_position, residues in item.hypothesis.proposed_residues.items():
            by_position[int(raw_position)][item.channel] = tuple(residues)
    conflicts: list[CrossChannelConflict] = []
    for position, channel_residues in sorted(by_position.items()):
        distinct = {tuple(value) for value in channel_residues.values()}
        if len(channel_residues) > 1 and len(distinct) > 1:
            conflicts.append(
                CrossChannelConflict(
                    position=position,
                    channels=tuple(sorted(channel_residues)),
                    residue_sets={key: value for key, value in channel_residues.items()},
                )
            )
    return tuple(conflicts)


def _default_explanation(
    approved: tuple[ApprovedSubHypothesis, ...],
    conflicts: tuple[CrossChannelConflict, ...],
) -> dict[str, Any]:
    return {
        "summary": "Synthesis of independently reviewed channel hypotheses.",
        "channel_contributions": [
            {
                "channel": item.channel,
                "sub_hypothesis_id": item.hypothesis.sub_hypothesis_id,
                "claim": item.hypothesis.claim,
                "evidence_ids": list(item.hypothesis.evidence_ids),
                "uncertainty": item.hypothesis.uncertainty,
            }
            for item in approved
        ],
        "conflicts": [item.model_dump(mode="json") for item in conflicts],
        "limitations": [
            "Channel outputs are hypotheses, not measured fitness or validated mechanisms."
        ],
    }


class HypothesisReviewGraph:
    """Three parallel child branches followed by one isolated main synthesis gate."""

    def __init__(
        self,
        *,
        child_scientists: Mapping[ChannelName, Any],
        child_critics: Mapping[ChannelName, Any],
        main_critic: Any,
        required_channels: tuple[ChannelName, ...] = FEATURE_CHANNELS,
        max_parallel_branches: int = 3,
        max_child_revision_attempts: int = 1,
        max_main_revision_attempts: int = 1,
        partitioner: KGContextPartitioner | None = None,
    ) -> None:
        if max_parallel_branches not in {1, 2, 3}:
            raise ValueError("max_parallel_branches must be between 1 and 3")
        if max_child_revision_attempts not in {0, 1, 2}:
            raise ValueError("max_child_revision_attempts must be between 0 and 2")
        if max_main_revision_attempts not in {0, 1, 2}:
            raise ValueError("max_main_revision_attempts must be between 0 and 2")
        self.child_scientists = dict(child_scientists)
        self.child_critics = dict(child_critics)
        self.main_critic = main_critic
        self.required_channels = required_channels
        self.max_parallel_branches = max_parallel_branches
        self.max_child_revision_attempts = max_child_revision_attempts
        self.max_main_revision_attempts = max_main_revision_attempts
        self.partitioner = partitioner or KGContextPartitioner()

    def _run_branch(
        self,
        *,
        channel: ChannelName,
        base_context: ScientistContextInput,
        evidence: tuple[Evidence, ...],
        packs: tuple[Any, ...],
    ) -> BranchReceipt:
        usable_evidence = any(item.quality_status != "unavailable" for item in evidence)
        usable_pack = any(
            any(
                (
                    pack.facts,
                    pack.predictions,
                    pack.directional_signals,
                    pack.counterevidence,
                )
            )
            or any(
                str(item.get("quality_status", "ok")) != "unavailable"
                for item in pack.evidence
            )
            for pack in packs
        )
        if not usable_evidence and not usable_pack:
            return BranchReceipt(
                channel=channel,
                status="SKIPPED_UNAVAILABLE",
                attempts=0,
                error_code="CHANNEL_INPUT_UNAVAILABLE",
            )
        immutable_context = self.partitioner.child_context(
            base_context=base_context,
            channel=channel,
            evidence=evidence,
            packs=packs,
        )
        input_hash = canonical_sha256(
            immutable_context.model_dump(mode="json", exclude={"retry_control"})
        )
        retry_control = None
        last_code = "CHILD_REVIEW_EXHAUSTED"
        last_completion: dict[str, Any] = {}
        for attempt in range(self.max_child_revision_attempts + 1):
            context = self.partitioner.child_context(
                base_context=base_context,
                channel=channel,
                evidence=evidence,
                packs=packs,
                retry_control=retry_control,
            )
            try:
                reset_completion_receipt()
                hypothesis = self.child_scientists[channel].propose(context=context)
                last_completion = completion_receipt_snapshot()
                validate_channel_hypothesis(
                    hypothesis.model_dump(mode="json"), context=context
                )
                review = self.child_critics[channel].review(
                    context=context, hypothesis=hypothesis
                )
            except Exception as error:  # noqa: BLE001 - graph receipts capture role failures
                return BranchReceipt(
                    channel=channel,
                    status="FAILED",
                    attempts=attempt + 1,
                    **_failure_fields(
                        error,
                        completion=completion_receipt_snapshot() or last_completion,
                    ),
                )
            output_hash = canonical_sha256(hypothesis.model_dump(mode="json"))
            if review.verdict == "APPROVE":
                approved = ApprovedSubHypothesis(
                    channel=channel,
                    hypothesis=hypothesis,
                    review=review,
                    attempt=attempt,
                    input_sha256=input_hash,
                    output_sha256=output_hash,
                )
                return BranchReceipt(
                    channel=channel,
                    status="SUCCEEDED",
                    attempts=attempt + 1,
                    input_chars=last_completion.get("input_chars"),
                    failure_category=last_completion.get("failure_category"),
                    request_started=bool(last_completion.get("request_started", False)),
                    approved=approved,
                )
            if review.verdict == "REJECT":
                last_code = "CHILD_CRITIC_REJECTED"
                break
            last_code = "CHILD_CRITIC_REVISION_EXHAUSTED"
            retry_control = {
                "schema": "critic_retry_control.v1",
                "priority": "highest",
                "attempt": attempt + 1,
                "immutable_input_sha256": input_hash,
                "rejected_output_sha256": output_hash,
                "decision_id": review.decision_id,
                "issue_codes": [item.code for item in review.issues],
                "required_changes": list(review.required_changes),
                "critic_summary": review.summary,
            }
        return BranchReceipt(
            channel=channel,
            status="FAILED",
            attempts=self.max_child_revision_attempts + 1,
            error_code=last_code,
            input_chars=last_completion.get("input_chars"),
            failure_category="review",
            request_started=bool(last_completion.get("request_started", False)),
        )

    def run(
        self,
        *,
        base_context: ScientistContextInput | dict[str, Any],
        evidence: tuple[Evidence, ...] | list[Evidence],
        interaction: InteractionResult | None,
        main_proposer: MainProposer,
    ) -> HypothesisPipelineResult:
        try:
            context = ScientistContextInput.model_validate(base_context)
            channel_packs, base_interaction = self.partitioner.split_packs(interaction)
            channel_evidence, base_evidence = self.partitioner.split_evidence(evidence)
        except Exception as error:  # noqa: BLE001 - projection failures become terminal receipts
            return HypothesisPipelineResult(
                status="FAILED",
                branches=(),
                conflicts=(),
                failure_code=(
                    f"CONTEXT_PROJECTION_FAILED:{type(error).__name__}:{str(error)[:240]}"
                ),
            )
        receipts: list[BranchReceipt] = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_branches) as executor:
            future_channel = {
                executor.submit(
                    copy_context().run,
                    self._run_branch,
                    channel=channel,
                    base_context=context,
                    evidence=channel_evidence[channel],
                    packs=channel_packs[channel],
                ): channel
                for channel in FEATURE_CHANNELS
            }
            for future in as_completed(future_channel):
                channel = future_channel[future]
                try:
                    receipts.append(future.result())
                except Exception as error:  # noqa: BLE001 - fail closed at branch boundary
                    receipts.append(
                        BranchReceipt(
                            channel=channel,
                            status="FAILED",
                            attempts=0,
                            **_failure_fields(error),
                        )
                    )
        receipts.sort(key=lambda item: FEATURE_CHANNELS.index(item.channel))
        failed_required = [
            item.channel
            for item in receipts
            if item.channel in self.required_channels and item.status != "SUCCEEDED"
        ]
        if failed_required:
            return HypothesisPipelineResult(
                status="FAILED",
                branches=tuple(receipts),
                conflicts=(),
                failure_code=f"REQUIRED_CHILD_FAILED:{','.join(failed_required)}",
            )
        approved = tuple(
            item.approved
            for item in receipts
            if item.status == "SUCCEEDED" and item.approved is not None
        )
        conflicts = _conflicts(approved)
        allowed_ids = frozenset(item.evidence_id for item in base_evidence)
        revision = None
        for attempt in range(self.max_main_revision_attempts + 1):
            try:
                hypothesis = main_proposer(
                    approved_subhypotheses=approved,
                    cross_channel_conflicts=conflicts,
                    base_interaction=base_interaction,
                    base_evidence=base_evidence,
                    critic_revision=revision,
                    hypothesis_attempt=attempt,
                )
                if not hypothesis.explanation:
                    hypothesis = replace(
                        hypothesis,
                        explanation=_default_explanation(approved, conflicts),
                    )
                review = self.main_critic.review(
                    hypothesis=hypothesis,
                    approved=approved,
                    conflicts=conflicts,
                    allowed_evidence_ids=allowed_ids,
                )
            except Exception as error:  # noqa: BLE001 - graph must emit a terminal receipt
                return HypothesisPipelineResult(
                    status="FAILED",
                    branches=tuple(receipts),
                    conflicts=conflicts,
                    main_attempts=attempt + 1,
                    failure_code=f"MAIN_NODE_FAILED:{type(error).__name__}:{str(error)[:240]}",
                )
            if review.verdict == "APPROVE":
                return HypothesisPipelineResult(
                    status="SUCCEEDED",
                    branches=tuple(receipts),
                    conflicts=conflicts,
                    main_hypothesis=hypothesis.__dict__,
                    main_review=review,
                    main_attempts=attempt + 1,
                )
            if review.verdict == "REJECT":
                break
            revision = {
                "schema": "critic_retry_control.v1",
                "priority": "highest",
                "verdict": review.verdict,
                "rejected_hypothesis_id": hypothesis.hypothesis_id,
                "rejected_preferred_residues": {
                    str(key): list(value)
                    for key, value in hypothesis.preferred_residues.items()
                },
                "decision_id": review.decision_id,
                "issue_codes": [item.code for item in review.issues],
                "required_changes": list(review.required_changes),
                "summary": review.summary,
            }
        return HypothesisPipelineResult(
            status="FAILED",
            branches=tuple(receipts),
            conflicts=conflicts,
            main_attempts=self.max_main_revision_attempts + 1,
            failure_code="MAIN_CRITIC_NOT_APPROVED",
        )
