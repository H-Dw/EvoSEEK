from dataclasses import replace

import pytest

from fitness_agents.agents.output_guards import (
    ResidueSubstitutionConstraint,
    RevisionConstraints,
)
from fitness_agents.config import AgentQuotaAllocationConfig
from fitness_agents.contracts.schemas import DesignScore, Variant
from fitness_agents.mutation import AgentQuotaBatchAcquisition


def _variant(variant_id: str, code: str, mutation_count: int = 2) -> Variant:
    return Variant(variant_id, code, code, code, mutation_count, "pool")


def _score(
    variant_id: str,
    *,
    utility: float,
    hypothesis: float,
    evidence: float = 0.0,
    prior: float = 0.0,
    uncertainty: float = 0.1,
) -> DesignScore:
    return DesignScore(
        variant_id=variant_id,
        utility=utility,
        uncertainty=uncertainty,
        hypothesis_score=hypothesis,
        evidence_score=evidence,
        prior_score=prior,
        predictor_score=0.0,
        selection_driver="kg_llm_uq",
        reason="test",
    )


def test_agent_quota_acquisition_realizes_eight_three_three_two() -> None:
    target_codes = ("AAAA", "AAAC", "AAAG", "AAAT", "AACA", "AAGA", "ATAA", "CAAA")
    control_codes = ("AAAD", "AADA")
    evidence_codes = ("CCCC", "CCCD", "CCCE")
    coverage_codes = ("GGGG", "GGGH", "GGGI")
    variants = [
        *[_variant(f"h{index}", code) for index, code in enumerate(target_codes)],
        *[_variant(f"c{index}", code) for index, code in enumerate(control_codes)],
        *[_variant(f"e{index}", code, 3) for index, code in enumerate(evidence_codes)],
        *[_variant(f"u{index}", code, 4) for index, code in enumerate(coverage_codes)],
    ]
    scores = [
        *[
            _score(f"h{index}", utility=1.0 - index * 0.01, hypothesis=0.75)
            for index in range(8)
        ],
        *[
            _score(f"c{index}", utility=0.3, hypothesis=0.5)
            for index in range(2)
        ],
        *[
            _score(f"e{index}", utility=0.4, hypothesis=0.0, evidence=0.9 - index * 0.1)
            for index in range(3)
        ],
        *[
            _score(f"u{index}", utility=0.2, hypothesis=0.0, uncertainty=0.9 - index * 0.1)
            for index in range(3)
        ],
    ]
    acquisition = AgentQuotaBatchAcquisition(
        AgentQuotaAllocationConfig(enabled=True)
    )

    result = acquisition.select(variants, scores, 16, diversity_lambda=0.1)

    assert len(result.selected_ids) == 16
    assert {arm: len(ids) for arm, ids in result.selected_by_arm.items()} == {
        "hypothesis_target": 8,
        "evidence_prior": 3,
        "coverage_exploration": 3,
        "matched_control": 2,
    }
    assert result.shortfalls == {arm: 0 for arm in result.quotas}
    assert result.fallback_ids == ()
    assert set(result.selected_by_arm["matched_control"]) == {"c0", "c1"}
    assert set(result.matched_control_pairs) == {"c0", "c1"}
    assert set(result.matched_control_pairs.values()).issubset(
        result.selected_by_arm["hypothesis_target"]
    )


def test_agent_quota_acquisition_audits_shortfall_and_fills_batch() -> None:
    config = AgentQuotaAllocationConfig(
        enabled=True,
        hypothesis_target=2,
        evidence_prior=1,
        coverage_exploration=1,
        matched_control=0,
    )
    variants = [_variant("h", "AAAA"), _variant("u1", "CCCC"), _variant("u2", "GGGG")]
    scores = [
        _score("h", utility=1.0, hypothesis=1.0),
        _score("u1", utility=0.7, hypothesis=0.0, uncertainty=0.9),
        _score("u2", utility=0.6, hypothesis=0.0, uncertainty=0.8),
    ]

    result = AgentQuotaBatchAcquisition(config).select(
        variants, scores, 3, diversity_lambda=0.0
    )

    assert len(result.selected_ids) == 3
    assert result.shortfalls["hypothesis_target"] == 1
    assert result.shortfalls["evidence_prior"] == 1
    assert result.fallback_ids


def test_revision_constraints_filter_complete_variants_and_expose_control_intent() -> None:
    config = AgentQuotaAllocationConfig(
        enabled=True,
        hypothesis_target=2,
        evidence_prior=1,
        coverage_exploration=1,
        matched_control=1,
    )
    variants = [
        _variant("a1", "ADGV"),
        _variant("a2", "AAGV"),
        _variant("c1", "VDGV", 0),
        _variant("e1", "CDGV"),
        _variant("u1", "GDGV"),
    ]
    scores = [
        _score("a1", utility=1.0, hypothesis=0.9),
        _score("a2", utility=0.9, hypothesis=0.8),
        _score("c1", utility=0.2, hypothesis=0.2),
        _score("e1", utility=0.5, hypothesis=0.0, evidence=0.9),
        _score("u1", utility=0.4, hypothesis=0.0, uncertainty=0.9),
    ]
    constraints = RevisionConstraints(
        excluded_substitutions=(
            ResidueSubstitutionConstraint(
                position=39,
                from_residue="V",
                to_residue="A",
            ),
        ),
        applies_to_arms=(
            "hypothesis_target",
            "evidence_prior",
            "coverage_exploration",
            "matched_control",
            "fallback",
        ),
    )

    result = AgentQuotaBatchAcquisition(config).select(
        variants,
        scores,
        5,
        diversity_lambda=0.0,
        constraints=constraints,
        position_to_index={39: 0, 40: 1, 41: 2, 54: 3},
        wild_type_by_position={39: "V", 40: "D", 41: "G", 54: "V"},
    )

    assert not {"a1", "a2"}.intersection(result.selected_ids)
    assert result.eligible_after_filter == 3
    assert len(result.selected_ids) == 3
    assert result.constraint_excluded_ids == ("a1", "a2")

    unconstrained = AgentQuotaBatchAcquisition(config).select(
        variants, scores, 5, diversity_lambda=0.0
    )
    control_id = unconstrained.selected_by_arm["matched_control"][0]
    control = unconstrained.intent_by_id()[control_id]
    assert control.matched_to == unconstrained.matched_control_pairs[control_id]
    assert control.allow_hypothesis_mismatch is True


def test_quota_configuration_must_match_round_budget(experiment_config) -> None:
    generation = replace(
        experiment_config.generation,
        selection_driver="agent_uq",
        quota_allocation=AgentQuotaAllocationConfig(
            enabled=True,
            hypothesis_target=2,
            evidence_prior=1,
            coverage_exploration=1,
            matched_control=0,
        ),
    )
    replace(experiment_config, generation=generation)

    with pytest.raises(ValueError, match="sum to budget_per_round"):
        replace(
            experiment_config,
            generation=replace(
                generation,
                quota_allocation=AgentQuotaAllocationConfig(
                    enabled=True,
                    hypothesis_target=8,
                    evidence_prior=3,
                    coverage_exploration=3,
                    matched_control=2,
                ),
            ),
        )
