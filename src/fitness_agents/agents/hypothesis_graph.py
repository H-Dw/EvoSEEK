"""Explicit fan-out/fan-in Scientist--Critic graph with bounded revisions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from inspect import Parameter, signature
from typing import Any

from fitness_agents.contracts.agent_io import ScientistContextInput
from fitness_agents.contracts.evidence_universe import RoleVisibleEvidenceUniverse
from fitness_agents.contracts.hypothesis_pipeline import (
    ApprovedChannelAnalysis,
    BatchedChannelAnalysisResult,
    BranchReceipt,
    ChannelName,
    ChildReviewAttemptArtifact,
    CrossChannelConflict,
    HypothesisPipelineResult,
    MainReviewAttemptArtifact,
    SynthesisAbstention,
)
from fitness_agents.contracts.schemas import Evidence, Hypothesis
from fitness_agents.kg_interaction.contracts import InteractionResult

from .context_projection import (
    FEATURE_CHANNELS,
    KGContextPartitioner,
    main_synthesis_evidence_cards,
    select_main_review_evidence_cards,
)
from .remote_llm import completion_receipt_snapshot, reset_completion_receipt
from .subscientist import validate_channel_hypothesis

MainProposer = Callable[..., Hypothesis | SynthesisAbstention]


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
        "failure_stage": getattr(error, "failure_stage", None),
        "batch_id": getattr(error, "batch_id", None),
        "sample_ids": tuple(getattr(error, "sample_ids", ())),
        "validation_paths": tuple(getattr(error, "validation_paths", ())),
        "completed_artifacts": tuple(getattr(error, "completed_artifacts", ())),
    }


def _conflicts(
    approved: tuple[ApprovedChannelAnalysis, ...]
) -> tuple[CrossChannelConflict, ...]:
    by_position: dict[int, dict[ChannelName, tuple[str, ...]]] = defaultdict(dict)
    for item in approved:
        for candidate in item.hypothesis.candidate_hypotheses:
            for raw_position, residues in candidate.proposed_residues.items():
                existing = by_position[int(raw_position)].get(item.channel, ())
                by_position[int(raw_position)][item.channel] = tuple(
                    dict.fromkeys((*existing, *residues))
                )
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
        max_main_revision_attempts: int = 2,
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
        input_receipt_id = f"IN-{channel[:2].upper()}-R{immutable_context.round_id:02d}"
        retry_control = None
        last_code = "CHILD_REVIEW_EXHAUSTED"
        last_completion: dict[str, Any] = {}
        attempt_artifacts: list[ChildReviewAttemptArtifact] = []
        for attempt in range(self.max_child_revision_attempts + 1):
            context = self.partitioner.child_context(
                base_context=base_context,
                channel=channel,
                evidence=evidence,
                packs=packs,
                retry_control=retry_control,
            )
            branch_evidence_universe = RoleVisibleEvidenceUniverse.from_role_sources(
                role=f"subcritic:{channel}",
                evidence=context.evidence,
                interaction={"packs": context.kg_packs},
            )
            hypothesis = None
            analysis_batches = ()
            output_receipt_id = None
            try:
                reset_completion_receipt()
                proposal = self.child_scientists[channel].propose(context=context)
                if isinstance(proposal, BatchedChannelAnalysisResult):
                    hypothesis = proposal.analysis
                    analysis_batches = proposal.batches
                    batch_input_chars = [
                        item.input_chars
                        for item in analysis_batches
                        if item.input_chars is not None
                    ]
                    last_completion = {
                        "input_chars": max(batch_input_chars) if batch_input_chars else None,
                        "failure_category": None,
                        "request_started": any(
                            item.request_started for item in analysis_batches
                        ),
                    }
                else:
                    hypothesis = proposal
                    last_completion = completion_receipt_snapshot()
                validate_channel_hypothesis(
                    hypothesis.model_dump(mode="json"), context=context
                )
                output_receipt_id = (
                    f"OUT-{channel[:2].upper()}-R{context.round_id:02d}-A{attempt:02d}"
                )
                review = self.child_critics[channel].review(
                    context=context, hypothesis=hypothesis
                )
            except Exception as error:  # noqa: BLE001 - graph receipts capture role failures
                failure = _failure_fields(
                    error,
                    completion=completion_receipt_snapshot() or last_completion,
                )
                attempt_artifacts.append(
                    ChildReviewAttemptArtifact(
                        channel=channel,
                        attempt=attempt,
                        disposition="FAILED",
                        input_receipt_id=input_receipt_id,
                        evidence_universe=branch_evidence_universe,
                        output_receipt_id=output_receipt_id,
                        analysis=hypothesis,
                        analysis_batches=(
                            analysis_batches or failure["completed_artifacts"]
                        ),
                        error_code=failure["error_code"],
                        input_chars=failure["input_chars"],
                        request_started=failure["request_started"],
                        failure_stage=failure["failure_stage"],
                        failed_batch_id=failure["batch_id"],
                        failed_sample_ids=failure["sample_ids"],
                        validation_paths=failure["validation_paths"],
                    )
                )
                return BranchReceipt(
                    channel=channel,
                    status="FAILED",
                    attempts=attempt + 1,
                    review_attempts=tuple(attempt_artifacts),
                    **failure,
                )
            attempt_artifacts.append(
                ChildReviewAttemptArtifact(
                    channel=channel,
                    attempt=attempt,
                    disposition={
                        "APPROVE": "APPROVED",
                        "REVISE": "REVISE",
                        "REJECT": "REJECTED",
                    }[review.verdict],
                    input_receipt_id=input_receipt_id,
                    evidence_universe=branch_evidence_universe,
                    output_receipt_id=output_receipt_id,
                    analysis=hypothesis,
                    analysis_batches=analysis_batches,
                    review=review,
                    input_chars=last_completion.get("input_chars"),
                    request_started=bool(last_completion.get("request_started", False)),
                )
            )
            if review.verdict == "APPROVE":
                approved = ApprovedChannelAnalysis(
                    channel=channel,
                    analysis=hypothesis,
                    review=review,
                    attempt=attempt,
                    input_receipt_id=input_receipt_id,
                    output_receipt_id=output_receipt_id,
                )
                return BranchReceipt(
                    channel=channel,
                    status="SUCCEEDED",
                    attempts=attempt + 1,
                    input_chars=last_completion.get("input_chars"),
                    failure_category=last_completion.get("failure_category"),
                    request_started=bool(last_completion.get("request_started", False)),
                    review_attempts=tuple(attempt_artifacts),
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
                "immutable_input_receipt_id": input_receipt_id,
                "rejected_output_receipt_id": output_receipt_id,
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
            review_attempts=tuple(attempt_artifacts),
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
        evidence_universe = RoleVisibleEvidenceUniverse.from_role_sources(
            role="main_scientist_and_critic",
            evidence=base_evidence,
            interaction=base_interaction,
            approved_channel_analyses=approved,
        )
        all_main_evidence_cards = main_synthesis_evidence_cards(
            evidence=evidence,
            interaction=interaction,
            approved=approved,
        )
        evidence_universe_id = f"EU-R{context.round_id:02d}"
        main_review_attempts: list[MainReviewAttemptArtifact] = []
        last_hypothesis: Hypothesis | None = None
        last_review = None
        revision = None
        for attempt in range(self.max_main_revision_attempts + 1):
            hypothesis = None
            review = None
            selected_evidence_cards = ()
            input_receipt_id = f"MAIN-IN-R{context.round_id:02d}-A{attempt:02d}"
            try:
                proposal = main_proposer(
                    approved_subhypotheses=approved,
                    cross_channel_conflicts=conflicts,
                    base_interaction=base_interaction,
                    base_evidence=base_evidence,
                    critic_revision=revision,
                    hypothesis_attempt=attempt,
                )
                if isinstance(proposal, SynthesisAbstention):
                    selected_evidence_cards = select_main_review_evidence_cards(
                        proposal, all_main_evidence_cards
                    )
                    main_review_attempts.append(
                        MainReviewAttemptArtifact(
                            hypothesis_attempt=attempt,
                            disposition="ABSTAINED",
                            input_receipt_id=input_receipt_id,
                            output_receipt_id=f"MAIN-OUT-R{context.round_id:02d}-A{attempt:02d}",
                            evidence_universe_id=evidence_universe_id,
                            evidence_cards=selected_evidence_cards,
                            abstention=proposal,
                        )
                    )
                    return HypothesisPipelineResult(
                        status="FAILED",
                        branches=tuple(receipts),
                        conflicts=conflicts,
                        evidence_universe=evidence_universe,
                        main_abstention=proposal,
                        main_review_attempts=tuple(main_review_attempts),
                        main_attempts=attempt + 1,
                        failure_code="NO_SUPPORTED_HYPOTHESIS",
                    )
                hypothesis = proposal
                selected_evidence_cards = select_main_review_evidence_cards(
                    hypothesis, all_main_evidence_cards
                )
                critic_evidence_universe = RoleVisibleEvidenceUniverse.from_role_sources(
                    role="main_critic", evidence=selected_evidence_cards
                )
                review_parameters = signature(self.main_critic.review).parameters.values()
                review_kwargs = {
                    "hypothesis": hypothesis,
                    "approved": approved,
                    "conflicts": conflicts,
                    "evidence_universe": critic_evidence_universe,
                }
                if any(
                    item.name == "evidence_cards"
                    or item.kind is Parameter.VAR_KEYWORD
                    for item in review_parameters
                ):
                    review_kwargs["evidence_cards"] = selected_evidence_cards
                review = self.main_critic.review(**review_kwargs)
            except Exception as error:  # noqa: BLE001 - graph must emit a terminal receipt
                if hypothesis is not None:
                    last_hypothesis = hypothesis
                main_review_attempts.append(
                    MainReviewAttemptArtifact(
                        hypothesis_attempt=attempt,
                        disposition="FAILED",
                        input_receipt_id=input_receipt_id,
                        output_receipt_id=(
                            f"MAIN-OUT-R{context.round_id:02d}-A{attempt:02d}"
                            if hypothesis is not None
                            else None
                        ),
                        evidence_universe_id=evidence_universe_id,
                        evidence_cards=selected_evidence_cards,
                        hypothesis=(
                            hypothesis.__dict__ if hypothesis is not None else None
                        ),
                        error_code=f"{type(error).__name__}:{str(error)[:240]}",
                    )
                )
                return HypothesisPipelineResult(
                    status="FAILED",
                    branches=tuple(receipts),
                    conflicts=conflicts,
                    evidence_universe=evidence_universe,
                    main_hypothesis=(
                        last_hypothesis.__dict__ if last_hypothesis is not None else None
                    ),
                    main_review=last_review,
                    main_review_attempts=tuple(main_review_attempts),
                    main_attempts=attempt + 1,
                    failure_code=f"MAIN_NODE_FAILED:{type(error).__name__}:{str(error)[:240]}",
                )
            last_hypothesis = hypothesis
            last_review = review
            disposition = {
                "APPROVE": "APPROVED",
                "REVISE": "REVISE",
                "REJECT": "REJECTED",
            }[review.verdict]
            main_review_attempts.append(
                MainReviewAttemptArtifact(
                    hypothesis_attempt=attempt,
                    disposition=disposition,
                    input_receipt_id=input_receipt_id,
                    output_receipt_id=f"MAIN-OUT-R{context.round_id:02d}-A{attempt:02d}",
                    evidence_universe_id=evidence_universe_id,
                    evidence_cards=selected_evidence_cards,
                    hypothesis=hypothesis.__dict__,
                    review=review,
                )
            )
            if review.verdict == "APPROVE":
                return HypothesisPipelineResult(
                    status="SUCCEEDED",
                    branches=tuple(receipts),
                    conflicts=conflicts,
                    evidence_universe=evidence_universe,
                    main_hypothesis=hypothesis.__dict__,
                    main_review=review,
                    main_review_attempts=tuple(main_review_attempts),
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
                "explanation": review.explanation,
            }
        return HypothesisPipelineResult(
            status="FAILED",
            branches=tuple(receipts),
            conflicts=conflicts,
            evidence_universe=evidence_universe,
            main_hypothesis=(
                last_hypothesis.__dict__ if last_hypothesis is not None else None
            ),
            main_review=last_review,
            main_review_attempts=tuple(main_review_attempts),
            main_attempts=len(main_review_attempts),
            failure_code="MAIN_CRITIC_NOT_APPROVED",
        )
