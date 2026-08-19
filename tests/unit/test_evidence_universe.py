from __future__ import annotations

from fitness_agents.contracts.evidence_universe import RoleVisibleEvidenceUniverse
from fitness_agents.contracts.hypothesis_pipeline import ChannelHypothesisOutput
from fitness_agents.contracts.schemas import Evidence
from fitness_agents.kg_interaction.contracts import EvidencePack, InteractionResult


def test_rag_id_is_allowed_by_exact_visibility_not_prefix() -> None:
    interaction = InteractionResult(
        plan_id="plan",
        packs=(
            EvidencePack(
                query_id="q:rag",
                operator="query_local_rag",
                as_of_round=1,
                evidence=(
                    {
                        "evidence_id": "ev:local_rag:visible",
                        "statement": "Visible RAG claim.",
                    },
                ),
            ),
        ),
        executed_steps=("rag",),
        skipped_steps=(),
        stop_reason="complete",
    )
    universe = RoleVisibleEvidenceUniverse.from_role_sources(
        role="main_critic", interaction=interaction
    )
    assert universe.require_known(["ev:local_rag:visible"]) == ()
    assert universe.require_known(["ev:local_rag:not-visible"]) == (
        "ev:local_rag:not-visible",
    )
    assert universe.entries[0].origins == ("kg_pack:query_local_rag",)


def test_universe_unions_evidence_kg_and_approved_analysis_ids() -> None:
    evidence = Evidence(
        evidence_id="ev:base",
        variant_id="v1",
        channel="history",
        statement="Visible observation.",
        score=0.1,
        source_id="source",
        confidence=0.5,
        round_id=1,
    )
    analysis = ChannelHypothesisOutput(
        analysis_id="analysis:pc:1",
        channel="physchem",
        analysis_summary="Bounded analysis.",
        findings=[
            {
                "finding_id": "finding:1",
                "kind": "OBSERVATION",
                "statement": "Visible finding.",
                "evidence_ids": ["ev:child"],
                "confidence": "medium",
            }
        ],
        evidence_ids=["ev:child"],
        counterevidence=[],
        uncertainty="Bounded uncertainty.",
    )
    universe = RoleVisibleEvidenceUniverse.from_role_sources(
        role="main",
        evidence=[evidence],
        approved_channel_analyses=[{"channel": "physchem", "hypothesis": analysis}],
    )
    assert universe.ids == frozenset({"ev:base", "ev:child"})
