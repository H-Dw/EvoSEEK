from __future__ import annotations

from threading import Barrier

from fitness_agents.agents.context_projection import KGContextPartitioner
from fitness_agents.agents.hypothesis_graph import HypothesisReviewGraph
from fitness_agents.agents.main_hypothesis_critic import RuleBasedMainHypothesisCritic
from fitness_agents.agents.subcritic import RuleBasedSubCritic
from fitness_agents.contracts.agent_io import ScientistContextInput
from fitness_agents.contracts.hypothesis_pipeline import (
    ChannelHypothesisOutput,
    HypothesisReviewIssue,
    HypothesisReviewOutput,
)
from fitness_agents.contracts.schemas import Evidence, Hypothesis
from fitness_agents.kg_interaction.contracts import EvidencePack, InteractionResult

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
            sub_hypothesis_id=f"sub:{context.channel}:{len(self.contexts)}",
            channel=context.channel,
            claim=f"Bounded {context.channel} claim.",
            proposed_residues={"39": ["V"]},
            evidence_ids=[f"ev:{context.channel}"],
            expected_effect="Test against a matched control.",
            counterevidence=[],
            uncertainty="Channel-local evidence does not establish fitness.",
            falsification_criterion="Revise if matched measurements oppose the direction.",
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
    assert result.main_hypothesis["explanation"]["channel_contributions"]


class _ReviseOnceCritic(RuleBasedSubCritic):
    def __init__(self) -> None:
        self.calls = 0

    def review(self, *, context, hypothesis):
        self.calls += 1
        if self.calls == 1:
            return HypothesisReviewOutput(
                decision_id="revise:1",
                verdict="REVISE",
                issues=[
                    HypothesisReviewIssue(
                        code="UNSUPPORTED_CLAIM",
                        severity="error",
                        message="Narrow the first draft.",
                    )
                ],
                required_changes=["NARROW_CLAIM"],
                cited_evidence_ids=[],
                summary="Narrow the channel-local claim.",
            )
        return super().review(context=context, hypothesis=hypothesis)


class _ReviseMainOnceCritic(RuleBasedMainHypothesisCritic):
    def __init__(self) -> None:
        self.calls = 0

    def review(self, *, hypothesis, approved, conflicts, allowed_evidence_ids):
        self.calls += 1
        if self.calls == 1:
            return HypothesisReviewOutput(
                decision_id="main:revise:1",
                verdict="REVISE",
                issues=[
                    HypothesisReviewIssue(
                        code="OVERCONFIDENT",
                        severity="error",
                        message="Calibrate the synthesis claim.",
                    )
                ],
                required_changes=["LOWER_CONFIDENCE"],
                cited_evidence_ids=[],
                summary="Lower confidence and preserve uncertainty.",
            )
        return super().review(
            hypothesis=hypothesis,
            approved=approved,
            conflicts=conflicts,
            allowed_evidence_ids=allowed_evidence_ids,
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
    assert retry["required_changes"] == ["NARROW_CLAIM"]


def test_main_critic_feedback_has_protected_priority_and_one_revision() -> None:
    main_calls = []

    def proposer(**kwargs):
        main_calls.append(kwargs)
        return _main_proposer(**kwargs)

    graph = HypothesisReviewGraph(
        child_scientists={channel: _Scientist() for channel in CHANNELS},
        child_critics={channel: RuleBasedSubCritic() for channel in CHANNELS},
        main_critic=_ReviseMainOnceCritic(),
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
