from __future__ import annotations

from fitness_agents.config import GenerationConfig
from fitness_agents.contracts.schemas import (
    CampaignState,
    Evidence,
    Hypothesis,
    Variant,
)
from fitness_agents.mutation.generators import KnowledgeCandidateGenerator
from fitness_agents.mutation.uncertainty import AgentUncertaintySelector

POSITIONS = {39: 0, 40: 1, 41: 2, 54: 3}
WILD_TYPE = {39: "V", 40: "D", 41: "G", 54: "V"}


def _variant(variant_id: str, code: str) -> Variant:
    return Variant(
        variant_id=variant_id,
        variant=code,
        sequence=code,
        mutation_notation="masked",
        mutation_count=sum(
            actual != native for actual, native in zip(code, "VDGV", strict=True)
        ),
        split_role="oracle_pool",
    )


def _hypothesis(preferred: dict[int, tuple[str, ...]]) -> Hypothesis:
    return Hypothesis(
        hypothesis_id="H1",
        statement="Test a bounded directional prior.",
        preferred_residues=preferred,
        evidence_ids=(),
        expected_outcome="The wet batch resolves the direction.",
        falsification_criterion="The registered batch comparison resolves the claim.",
    )


def _state() -> CampaignState:
    return CampaignState(
        run_id="RUN1",
        mode="knowledge_agent",
        seed=11,
        round_id=2,
    )


def _generator() -> KnowledgeCandidateGenerator:
    return KnowledgeCandidateGenerator(
        POSITIONS,
        sampling_namespace="fold2-guard",
        wild_type_by_position=WILD_TYPE,
    )


def _evidence(variant_id: str, *, selectable: bool) -> Evidence:
    return Evidence(
        evidence_id=f"E-{variant_id}",
        variant_id=variant_id,
        channel="local_rag",
        statement="An explanation-only boundary record.",
        score=1.0,
        source_id="synthetic-boundary",
        confidence=0.9,
        round_id=2,
        evidence_type="retrieved_logic_unit",
        raw_features={"permission": "explanation_only"},
        applicability="generic_or_other_protein_context",
        contributes_to_selection=selectable,
        warnings=("external_permission_explanation_only",),
    )


def test_wild_type_soft_preferences_do_not_reward_unedited_sites() -> None:
    low_depth = _variant("low", "VDGG")
    deep = _variant("deep", "IILA")
    hypothesis = _hypothesis(
        {
            39: ("V", "I"),
            40: ("D", "I"),
            41: ("G", "L"),
            54: ("V", "G"),
        }
    )

    selected = _generator().generate(
        (low_depth, deep),
        _state(),
        hypothesis,
        evidence={},
        limit=1,
    )

    assert [item.variant_id for item in selected] == ["deep"]


def test_wild_type_only_hypothesis_is_candidate_pool_neutral() -> None:
    candidates = tuple(
        _variant(variant_id, code)
        for variant_id, code in (
            ("v1", "VDGG"),
            ("v2", "IDGV"),
            ("v3", "VLGV"),
            ("v4", "IILA"),
            ("v5", "ADAV"),
            ("v6", "VDGV"),
        )
    )
    wild_type_only = _hypothesis(
        {39: ("V",), 40: ("D",), 41: ("G",), 54: ("V",)}
    )
    neutral = _hypothesis({})

    with_wild_type = _generator().generate(
        candidates,
        _state(),
        wild_type_only,
        evidence={},
        limit=4,
    )
    without_preferences = _generator().generate(
        candidates,
        _state(),
        neutral,
        evidence={},
        limit=4,
    )

    assert [item.variant_id for item in with_wild_type] == [
        item.variant_id for item in without_preferences
    ]


def test_explanation_only_boundary_evidence_cannot_prefilter_pool() -> None:
    candidates = tuple(
        _variant(variant_id, code)
        for variant_id, code in (
            ("v1", "VDGG"),
            ("v2", "IDGV"),
            ("v3", "VLGV"),
            ("v4", "IILA"),
        )
    )
    advisory = {
        "v1": [_evidence("v1", selectable=False)],
        "v2": [_evidence("v2", selectable=False)],
    }

    baseline = _generator().generate(
        candidates,
        _state(),
        hypothesis=None,
        evidence={},
        limit=2,
    )
    with_advisory = _generator().generate(
        candidates,
        _state(),
        hypothesis=None,
        evidence=advisory,
        limit=2,
    )
    with_authorized = _generator().generate(
        candidates,
        _state(),
        hypothesis=None,
        evidence={
            "v1": [_evidence("v1", selectable=True)],
            "v2": [_evidence("v2", selectable=True)],
        },
        limit=2,
    )

    assert [item.variant_id for item in with_advisory] == [
        item.variant_id for item in baseline
    ]
    assert {item.variant_id for item in with_authorized} == {"v1", "v2"}


def test_agent_uq_uses_the_same_edited_site_hypothesis_semantics() -> None:
    candidates = (
        _variant("wt", "VDGV"),
        _variant("low", "VDGG"),
        _variant("deep", "IILA"),
    )
    hypothesis = _hypothesis(
        {
            39: ("V", "I"),
            40: ("D", "I"),
            41: ("G", "L"),
            54: ("V", "G"),
        }
    )
    selector = AgentUncertaintySelector(
        GenerationConfig(
            hypothesis_weight=1.0,
            evidence_weight=0.0,
            prior_weight=0.0,
            uncertainty_beta=0.0,
        ),
        position_to_index=POSITIONS,
        wild_type_by_position=WILD_TYPE,
    )

    scores = {
        item.variant_id: item.hypothesis_score
        for item in selector.score(
            candidates,
            observed_variants=(candidates[0],),
            hypothesis=hypothesis,
            evidence={},
        )
    }

    assert scores == {"wt": 0.0, "low": 0.25, "deep": 0.75}
