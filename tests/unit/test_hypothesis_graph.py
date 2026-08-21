from __future__ import annotations

from threading import Barrier

import pytest

from fitness_agents.agents.context_projection import (
    KGContextPartitioner,
    approved_analysis_payload,
)
from fitness_agents.agents.hypothesis_graph import HypothesisReviewGraph
from fitness_agents.agents.main_hypothesis_critic import RuleBasedMainHypothesisCritic
from fitness_agents.agents.output_guards import SemanticOutputValidationError
from fitness_agents.agents.remote_llm import RemoteLLMCompletionError, complete_json
from fitness_agents.agents.subcritic import RuleBasedSubCritic
from fitness_agents.agents.subscientist import validate_channel_hypothesis
from fitness_agents.contracts.agent_io import ScientistContextInput
from fitness_agents.contracts.evidence_universe import RoleVisibleEvidenceUniverse
from fitness_agents.contracts.hypothesis_pipeline import (
    BatchedChannelAnalysisResult,
    ChannelAnalysisBatchArtifact,
    ChannelHypothesisOutput,
    MainReviewIssue,
    MainReviewOutput,
    PhyschemReviewIssue,
    PhyschemReviewOutput,
    SynthesisAbstention,
)
from fitness_agents.contracts.schemas import Evidence, Hypothesis
from fitness_agents.kg_interaction.contracts import EvidencePack, InteractionResult
from fitness_agents.utils.progress import bind_progress, report_event, reset_progress

CHANNELS = ("physchem", "conservation", "structure")


def _context() -> ScientistContextInput:
    return ScientistContextInput.model_validate(
        {
            "run_id": "run",
            "mode": "knowledge_agent",
            "round_id": 1,
            "expected_hypothesis_id": "hyp:run:r1",
            "task": "maximize visible assay fitness",
            "protein_id": "GB1",
            "objective": "maximize",
            "mutable_positions": [39, 40, 41, 54],
            "wild_type_sites": "VDGV",
            "protein_context_id": "ctx:test",
            "visible_observations": [],
            "previous_hypothesis_id": None,
            "previous_hypothesis_assessment": None,
        }
    )


def _evidence(channel: str) -> Evidence:
    return Evidence(
        evidence_id=f"ev:{channel}",
        variant_id="v1",
        channel=channel,
        statement=f"Visible {channel} statement.",
        score=0.1,
        source_id=f"source:{channel}",
        confidence=0.7,
        round_id=1,
    )


def _interaction() -> InteractionResult:
    return InteractionResult(
        plan_id="plan",
        packs=(
            EvidencePack(
                query_id="q:bundle",
                operator="query_feature_bundle",
                as_of_round=1,
                evidence=tuple(
                    {
                        "evidence_id": f"ev:{channel}",
                        "channel": channel,
                        "statement": f"Visible {channel} statement.",
                    }
                    for channel in CHANNELS
                ),
            ),
            EvidencePack(
                query_id="q:base",
                operator="hypothesis_context",
                as_of_round=1,
                facts=({"fact_type": "visible_observation_count", "value": 0},),
            ),
        ),
        executed_steps=("bundle", "base"),
        skipped_steps=(),
        stop_reason="complete",
    )


class _Scientist:
    def __init__(self, barrier: Barrier | None = None) -> None:
        self.barrier = barrier
        self.contexts = []

    def propose(self, *, context):
        self.contexts.append(context)
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        return ChannelHypothesisOutput(
            analysis_id=f"analysis:{context.channel}:{len(self.contexts)}",
            channel=context.channel,
            analysis_summary=f"Bounded {context.channel} analysis.",
            findings=[
                {
                    "finding_id": f"finding:{context.channel}:1",
                    "kind": "OBSERVATION",
                    "statement": f"Visible {context.channel} observation.",
                    "evidence_ids": [f"ev:{context.channel}"],
                    "confidence": "medium",
                }
            ],
            candidate_hypotheses=[
                {
                    "hypothesis_id": f"candidate:{context.channel}:1",
                    "statement": f"Bounded {context.channel} candidate hypothesis.",
                    "proposed_residues": {"39": ["V"]},
                    "evidence_ids": [f"ev:{context.channel}"],
                    "expected_observation": "Test against a matched control.",
                    "falsification_criterion": (
                        "Revise if matched measurements oppose the direction."
                    ),
                }
            ],
            evidence_ids=[f"ev:{context.channel}"],
            counterevidence=[],
            uncertainty="Channel-local evidence does not establish fitness.",
        )


class _BatchedScientist(_Scientist):
    def propose(self, *, context):
        analysis = super().propose(context=context)
        return BatchedChannelAnalysisResult(
            analysis=analysis,
            batches=(
                ChannelAnalysisBatchArtifact(
                    batch_id="b000",
                    split_depth=0,
                    sample_ids=("v1",),
                    input_receipt_id="IN-01",
                    output_receipt_id="OUT-01",
                    evidence_universe=RoleVisibleEvidenceUniverse.from_role_sources(
                        role=f"subscientist:{context.channel}",
                        evidence=context.evidence,
                        interaction={"packs": context.kg_packs},
                    ),
                    analysis=analysis,
                    input_chars=1000,
                    request_started=True,
                ),
            ),
        )


def _main_proposer(**kwargs):
    assert [item.channel for item in kwargs["approved_subhypotheses"]] == list(CHANNELS)
    # Raw feature packs were consumed by children and cannot reach the main role.
    assert [pack.operator for pack in kwargs["base_interaction"].packs] == [
        "hypothesis_context"
    ]
    assert kwargs["base_evidence"] == ()
    return Hypothesis(
        hypothesis_id="hyp:run:r1",
        statement="Synthesize the three independently reviewed channel directions.",
        preferred_residues={39: ("V",), 40: ("D",), 41: ("G",), 54: ("V",)},
        evidence_ids=tuple(f"ev:{channel}" for channel in CHANNELS),
        expected_outcome="Tested variants should differ from matched controls.",
        falsification_criterion="Revise if the matched comparison opposes the direction.",
    )


def test_context_partitioner_prevents_cross_channel_visibility_and_deduplicates() -> None:
    partitioner = KGContextPartitioner()
    packs, base = partitioner.split_packs(_interaction())
    assert [pack.operator for pack in base.packs] == ["hypothesis_context"]
    for channel in CHANNELS:
        visible = packs[channel][0].evidence
        assert {item["channel"] for item in visible} == {channel}
        assert {item["evidence_id"] for item in visible} == {f"ev:{channel}"}
        child = partitioner.child_context(
            base_context=_context(),
            channel=channel,
            evidence=(_evidence(channel),),
            packs=packs[channel],
        )
        visible_occurrences = sum(
            item.get("evidence_id") == f"ev:{channel}" for item in child.evidence
        ) + sum(
            item.get("evidence_id") == f"ev:{channel}"
            for pack in child.kg_packs
            for item in pack.get("evidence", ())
        )
        assert visible_occurrences == 1


def test_feature_child_sample_card_is_fitness_blind_and_channel_typed() -> None:
    raw_context = _context().model_dump(mode="json")
    raw_context["visible_observations"] = [
        {
            "variant_id": "v1",
            "variant": "ADGV",
            "mutation_notation": "V39A",
            "residues_by_position": {"39": "A", "40": "D", "41": "G", "54": "V"},
            "measured_fitness": 99.0,
        }
    ]
    evidence = Evidence(
        **{
            **_evidence("physchem").__dict__,
            "raw_features": {
                "sites": {
                    "39": {
                        "mutation": "V39A",
                        "deltas": {"nominal_charge": 1.0},
                    }
                }
            },
        }
    )
    child = KGContextPartitioner().child_context(
        base_context=ScientistContextInput.model_validate(raw_context),
        channel="physchem",
        evidence=(evidence,),
        packs=(),
    )
    payload = child.model_dump(mode="json")
    serialized = str(payload["visible_observations"])
    assert "measured_fitness" not in serialized
    assert payload["visible_observations"][0]["sample_id"] == "S01"
    assert payload["sample_map"] == {"S01": "V39A"}
    assert payload["visible_observations"][0]["evidence_ids"] == ["ev:physchem"]
    assert (
        payload["visible_observations"][0]["feature_values"]["ev:physchem"]["kind"]
        == "physchem"
    )
    fact = child.visible_observations[0].descriptor_facts[0]
    assert (
        fact.sample_id,
        fact.position,
        fact.from_residue,
        fact.to_residue,
        fact.delta,
    ) == ("S01", 39, "V", "A", 1.0)

    valid = ChannelHypothesisOutput(
        analysis_id="analysis:physchem:fact-binding",
        channel="physchem",
        analysis_summary="One typed descriptor delta is visible.",
        findings=[
            {
                "finding_id": "finding:physchem:fact-binding",
                "kind": "OBSERVATION",
                "statement": "Nominal charge delta is 1 for V39A in sample v1.",
                "evidence_ids": ["ev:physchem"],
                "fact_ids": [fact.fact_id],
                "confidence": "medium",
            }
        ],
        candidate_hypotheses=[],
        evidence_ids=["ev:physchem"],
        fact_ids=[fact.fact_id],
        counterevidence=[],
        uncertainty="A descriptor delta is not a fitness measurement.",
    )
    validate_channel_hypothesis(valid.model_dump(mode="json"), context=child)

    mismatched = valid.model_copy(
        update={
            "findings": [
                valid.findings[0].model_copy(
                    update={"statement": "Nominal charge delta is 1 for G41D."}
                )
            ]
        }
    )
    with pytest.raises(SemanticOutputValidationError, match="match the cited mutation"):
        validate_channel_hypothesis(mismatched.model_dump(mode="json"), context=child)


def test_three_child_branches_execute_in_parallel_and_main_gets_only_approved_summaries() -> None:
    barrier = Barrier(3)
    scientists = {channel: _Scientist(barrier) for channel in CHANNELS}
    graph = HypothesisReviewGraph(
        child_scientists=scientists,
        child_critics={channel: RuleBasedSubCritic() for channel in CHANNELS},
        main_critic=RuleBasedMainHypothesisCritic(),
    )
    result = graph.run(
        base_context=_context(),
        evidence=[_evidence(channel) for channel in CHANNELS],
        interaction=_interaction(),
        main_proposer=_main_proposer,
    )
    assert result.status == "SUCCEEDED"
    assert all(item.status == "SUCCEEDED" for item in result.branches)
    assert result.main_review is not None
    assert result.main_review.verdict == "APPROVE"
    assert result.main_review_attempts[-1].disposition == "APPROVED"
    assert len(result.main_review_attempts[-1].evidence_cards) <= 12
    assert "explanation" not in result.main_hypothesis
    assert result.main_review.explanation
    assert all(len(item.review_attempts) == 1 for item in result.branches)
    assert all(item.review_attempts[0].disposition == "APPROVED" for item in result.branches)
    assert result.evidence_universe is not None
    assert "ev:physchem" in result.evidence_universe.ids
    assert all(
        item.review_attempts[0].evidence_universe.ids == frozenset({f"ev:{item.channel}"})
        for item in result.branches
    )
    main_payload = approved_analysis_payload(result.branches[0].approved)
    assert set(main_payload) == {
        "channel",
        "contribution_modes",
        "analysis",
        "semantic_review",
    }
    assert "input_sha256" not in str(main_payload)
    assert "decision_id" not in str(main_payload)
    serialized_approved = result.model_dump(mode="json")["branches"][0]["approved"]
    assert "analysis" in serialized_approved
    assert "hypothesis" not in serialized_approved


def test_typed_main_abstention_is_retained_without_calling_main_critic() -> None:
    class _NeverReview:
        def review(self, **kwargs):
            raise AssertionError(f"main critic must not review abstention: {kwargs}")

    graph = HypothesisReviewGraph(
        child_scientists={channel: _Scientist() for channel in CHANNELS},
        child_critics={channel: RuleBasedSubCritic() for channel in CHANNELS},
        main_critic=_NeverReview(),
    )

    def abstain(**kwargs):
        del kwargs
        return SynthesisAbstention(
            abstention_id="abstain:run:r1",
            reason="No visible card supports a residue direction.",
            evidence_ids=("ev:physchem",),
            unresolved_constraints=("All channel contributions are analysis-only.",),
            recommended_next_evidence=("Obtain a directional assay association.",),
        )

    result = graph.run(
        base_context=_context(),
        evidence=[_evidence(channel) for channel in CHANNELS],
        interaction=_interaction(),
        main_proposer=abstain,
    )
    assert result.status == "FAILED"
    assert result.failure_code == "NO_SUPPORTED_HYPOTHESIS"
    assert result.main_abstention is not None
    assert result.main_review_attempts[-1].disposition == "ABSTAINED"


def test_graph_persists_typed_subscientist_batch_artifacts_without_forwarding_them() -> None:
    graph = HypothesisReviewGraph(
        child_scientists={channel: _BatchedScientist() for channel in CHANNELS},
        child_critics={channel: RuleBasedSubCritic() for channel in CHANNELS},
        main_critic=RuleBasedMainHypothesisCritic(),
    )

    result = graph.run(
        base_context=_context(),
        evidence=[_evidence(channel) for channel in CHANNELS],
        interaction=_interaction(),
        main_proposer=_main_proposer,
    )

    assert result.status == "SUCCEEDED"
    assert all(
        item.review_attempts[0].analysis_batches[0].sample_ids == ("v1",)
        for item in result.branches
    )
    assert "analysis_batches" not in result.main_review.explanation


def test_main_critic_accepts_exact_rag_id_visible_in_base_interaction() -> None:
    base = _interaction()
    interaction = InteractionResult(
        plan_id=base.plan_id,
        packs=base.packs
        + (
            EvidencePack(
                query_id="q:rag",
                operator="query_local_rag",
                as_of_round=1,
                evidence=(
                    {
                        "evidence_id": "ev:local_rag:visible",
                        "statement": "Visible RAG statement.",
                    },
                ),
            ),
        ),
        executed_steps=base.executed_steps + ("rag",),
        skipped_steps=base.skipped_steps,
        stop_reason=base.stop_reason,
    )

    def proposer(**kwargs):
        assert any(
            pack.operator == "query_local_rag" for pack in kwargs["base_interaction"].packs
        )
        return Hypothesis(
            hypothesis_id="hyp:run:r1",
            statement="Synthesize reviewed analysis cards with visible RAG evidence.",
            preferred_residues={39: ("V",), 40: ("D",), 41: ("G",), 54: ("V",)},
            evidence_ids=(
                *(f"ev:{channel}" for channel in CHANNELS),
                "ev:local_rag:visible",
            ),
            expected_outcome="Tested variants should differ from matched controls.",
            falsification_criterion="Revise if the matched comparison opposes the direction.",
        )

    graph = HypothesisReviewGraph(
        child_scientists={channel: _Scientist() for channel in CHANNELS},
        child_critics={channel: RuleBasedSubCritic() for channel in CHANNELS},
        main_critic=RuleBasedMainHypothesisCritic(),
    )
    result = graph.run(
        base_context=_context(),
        evidence=[_evidence(channel) for channel in CHANNELS],
        interaction=interaction,
        main_proposer=proposer,
    )
    assert result.status == "SUCCEEDED"
    assert "ev:local_rag:visible" in result.main_hypothesis["evidence_ids"]


def test_child_threads_inherit_progress_context() -> None:
    class _Capture:
        def __init__(self) -> None:
            self.events = []

        def heartbeat(self, message, *, log=True, **payload):
            del message, log, payload

        def report(self, event_type, *, message, persist=True, **payload):
            del message, persist, payload
            self.events.append(event_type)

    class _ReportingScientist(_Scientist):
        def propose(self, *, context):
            report_event(f"child:{context.channel}", message="child event")
            return super().propose(context=context)

    capture = _Capture()
    token = bind_progress(capture)
    try:
        graph = HypothesisReviewGraph(
            child_scientists={channel: _ReportingScientist() for channel in CHANNELS},
            child_critics={channel: RuleBasedSubCritic() for channel in CHANNELS},
            main_critic=RuleBasedMainHypothesisCritic(),
        )
        result = graph.run(
            base_context=_context(),
            evidence=[_evidence(channel) for channel in CHANNELS],
            interaction=_interaction(),
            main_proposer=_main_proposer,
        )
    finally:
        reset_progress(token)

    assert result.status == "SUCCEEDED"
    assert set(capture.events) == {f"child:{channel}" for channel in CHANNELS}


def test_branch_receipt_preserves_structured_remote_failure() -> None:
    class _BudgetFailureScientist:
        def propose(self, *, context):
            del context
            raise RemoteLLMCompletionError(
                "PROMPT_BUDGET_EXCEEDED",
                failure_category="budget",
                input_chars=123456,
                request_started=False,
                detail="projected prompt exceeded the role budget",
            )

    scientists = {channel: _Scientist() for channel in CHANNELS}
    scientists["physchem"] = _BudgetFailureScientist()
    graph = HypothesisReviewGraph(
        child_scientists=scientists,
        child_critics={channel: RuleBasedSubCritic() for channel in CHANNELS},
        main_critic=RuleBasedMainHypothesisCritic(),
    )
    result = graph.run(
        base_context=_context(),
        evidence=[_evidence(channel) for channel in CHANNELS],
        interaction=_interaction(),
        main_proposer=_main_proposer,
    )

    receipt = next(item for item in result.branches if item.channel == "physchem")
    assert receipt.status == "FAILED"
    assert receipt.error_code == "PROMPT_BUDGET_EXCEEDED"
    assert receipt.input_chars == 123456
    assert receipt.failure_category == "budget"
    assert receipt.request_started is False


def test_successful_branch_receipt_records_started_request_size() -> None:
    class _Completions:
        def create(self, **kwargs):
            del kwargs
            message = type("Message", (), {"content": '{"ok": true}'})()
            choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
            return type("Response", (), {"choices": [choice], "usage": None})()

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    class _RemoteSizedScientist(_Scientist):
        def propose(self, *, context):
            complete_json(
                client=_Client(),
                model="unit-test-model",
                messages=[{"role": "user", "content": "json"}],
            )
            return super().propose(context=context)

    graph = HypothesisReviewGraph(
        child_scientists={channel: _RemoteSizedScientist() for channel in CHANNELS},
        child_critics={channel: RuleBasedSubCritic() for channel in CHANNELS},
        main_critic=RuleBasedMainHypothesisCritic(),
    )
    result = graph.run(
        base_context=_context(),
        evidence=[_evidence(channel) for channel in CHANNELS],
        interaction=_interaction(),
        main_proposer=_main_proposer,
    )

    assert result.status == "SUCCEEDED"
    assert all(item.input_chars == 4 for item in result.branches)
    assert all(item.request_started is True for item in result.branches)
    assert all(item.failure_category is None for item in result.branches)


class _ReviseOnceCritic(RuleBasedSubCritic):
    def __init__(self) -> None:
        self.calls = 0

    def review(self, *, context, hypothesis):
        self.calls += 1
        if self.calls == 1:
            return PhyschemReviewOutput(
                review_scope="physchem",
                decision_id="revise:1",
                verdict="REVISE",
                rating={
                    "score": 3,
                    "rationale": "The scope defect is repairable.",
                    "suggestions": ["Narrow the channel-local claim."],
                    "text_errors": [],
                },
                issues=[
                    PhyschemReviewIssue(
                        code="ANALYSIS_SCOPE_OVERREACH",
                        severity="error",
                        message="Narrow the first draft.",
                    )
                ],
                required_changes=["NARROW_ANALYSIS"],
                cited_evidence_ids=[],
                summary="Narrow the channel-local claim.",
            )
        return super().review(context=context, hypothesis=hypothesis)


class _ReviseMainOnceCritic(RuleBasedMainHypothesisCritic):
    def __init__(self) -> None:
        self.calls = 0
        self.review_kwargs: list[dict] = []

    def review(
        self,
        *,
        hypothesis,
        approved,
        conflicts,
        evidence_universe,
        prior_review=None,
        **kwargs,
    ):
        self.calls += 1
        self.review_kwargs.append({"prior_review": prior_review, **kwargs})
        if self.calls == 1:
            return MainReviewOutput(
                review_scope="main",
                decision_id="main:revise:1",
                verdict="REVISE",
                rating={
                    "score": 3,
                    "rationale": "The confidence defect is repairable.",
                    "suggestions": ["Lower confidence to match visible uncertainty."],
                    "text_errors": [],
                },
                issues=[
                    MainReviewIssue(
                        code="OVERCONFIDENT",
                        severity="error",
                        message="Calibrate the synthesis claim.",
                    )
                ],
                required_changes=["LOWER_CONFIDENCE"],
                cited_evidence_ids=[],
                explanation=(
                    "The exact hypothesis is testable, but its confidence exceeds the "
                    "visible uncertainty."
                ),
            )
        return super().review(
            hypothesis=hypothesis,
            approved=approved,
            conflicts=conflicts,
            evidence_universe=evidence_universe,
            prior_review=prior_review,
            **kwargs,
        )


def test_child_critic_feedback_is_structured_and_bounded_to_one_revision() -> None:
    scientists = {channel: _Scientist() for channel in CHANNELS}
    graph = HypothesisReviewGraph(
        child_scientists=scientists,
        child_critics={
            "physchem": _ReviseOnceCritic(),
            "conservation": RuleBasedSubCritic(),
            "structure": RuleBasedSubCritic(),
        },
        main_critic=RuleBasedMainHypothesisCritic(),
        max_child_revision_attempts=1,
    )
    result = graph.run(
        base_context=_context(),
        evidence=[_evidence(channel) for channel in CHANNELS],
        interaction=_interaction(),
        main_proposer=_main_proposer,
    )
    assert result.status == "SUCCEEDED"
    assert result.branches[0].attempts == 2
    retry = scientists["physchem"].contexts[1].retry_control
    assert retry["schema"] == "critic_retry_control.v1"
    assert retry["priority"] == "highest"
    assert retry["required_changes"] == ["NARROW_ANALYSIS"]
    assert retry["suggestions"] == ["Narrow the channel-local claim."]
    assert [item.disposition for item in result.branches[0].review_attempts] == [
        "REVISE",
        "APPROVED",
    ]


def test_main_critic_feedback_has_protected_priority_and_one_revision() -> None:
    main_calls = []

    def proposer(**kwargs):
        main_calls.append(kwargs)
        return _main_proposer(**kwargs)

    main_critic = _ReviseMainOnceCritic()
    graph = HypothesisReviewGraph(
        child_scientists={channel: _Scientist() for channel in CHANNELS},
        child_critics={channel: RuleBasedSubCritic() for channel in CHANNELS},
        main_critic=main_critic,
        max_main_revision_attempts=1,
    )
    result = graph.run(
        base_context=_context(),
        evidence=[_evidence(channel) for channel in CHANNELS],
        interaction=_interaction(),
        main_proposer=proposer,
    )
    assert result.status == "SUCCEEDED"
    assert result.main_attempts == 2
    retry = main_calls[1]["critic_revision"]
    assert retry["schema"] == "critic_retry_control.v1"
    assert retry["priority"] == "highest"
    assert retry["issue_codes"] == ["OVERCONFIDENT"]
    assert retry["required_changes"] == ["LOWER_CONFIDENCE"]
    assert retry["suggestions"] == ["Lower confidence to match visible uncertainty."]
    assert main_critic.review_kwargs[0]["prior_review"] is None
    prior = main_critic.review_kwargs[1]["prior_review"]
    assert prior["issue_codes"] == ["OVERCONFIDENT"]
    assert prior["required_changes"] == ["LOWER_CONFIDENCE"]
    assert prior["suggestions"] == ["Lower confidence to match visible uncertainty."]
    assert "explanation" not in prior
    assert all("message" not in item for item in prior["issues"])


def test_required_unavailable_channel_fails_before_main_synthesis() -> None:
    graph = HypothesisReviewGraph(
        child_scientists={channel: _Scientist() for channel in CHANNELS},
        child_critics={channel: RuleBasedSubCritic() for channel in CHANNELS},
        main_critic=RuleBasedMainHypothesisCritic(),
    )
    called = False

    def main(**kwargs):
        nonlocal called
        called = True
        return _main_proposer(**kwargs)

    result = graph.run(
        base_context=_context(),
        evidence=[_evidence("physchem"), _evidence("conservation")],
        interaction=None,
        main_proposer=main,
    )
    assert result.status == "FAILED"
    assert "structure" in result.failure_code
    assert called is False


def test_quality_unavailable_does_not_satisfy_required_channel() -> None:
    graph = HypothesisReviewGraph(
        child_scientists={channel: _Scientist() for channel in CHANNELS},
        child_critics={channel: RuleBasedSubCritic() for channel in CHANNELS},
        main_critic=RuleBasedMainHypothesisCritic(),
    )
    unavailable_structure = Evidence(
        evidence_id="ev:structure:unavailable",
        variant_id="v1",
        channel="structure",
        statement="Structure source unavailable.",
        score=0.0,
        source_id="source:structure",
        confidence=0.0,
        round_id=1,
        quality_status="unavailable",
    )
    result = graph.run(
        base_context=_context(),
        evidence=[
            _evidence("physchem"),
            _evidence("conservation"),
            unavailable_structure,
        ],
        interaction=None,
        main_proposer=_main_proposer,
    )
    assert result.status == "FAILED"
    assert "structure" in result.failure_code


def test_context_projection_exception_becomes_failed_receipt() -> None:
    class _BrokenPartitioner(KGContextPartitioner):
        def split_packs(self, interaction):
            del interaction
            raise ValueError("injected projection failure")

    graph = HypothesisReviewGraph(
        child_scientists={channel: _Scientist() for channel in CHANNELS},
        child_critics={channel: RuleBasedSubCritic() for channel in CHANNELS},
        main_critic=RuleBasedMainHypothesisCritic(),
        partitioner=_BrokenPartitioner(),
    )
    result = graph.run(
        base_context=_context(),
        evidence=[_evidence(channel) for channel in CHANNELS],
        interaction=_interaction(),
        main_proposer=_main_proposer,
    )
    assert result.status == "FAILED"
    assert result.failure_code.startswith("CONTEXT_PROJECTION_FAILED")
