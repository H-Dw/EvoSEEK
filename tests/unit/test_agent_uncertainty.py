from dataclasses import replace

import pytest

from fitness_agents.config import GenerationConfig
from fitness_agents.contracts.schemas import Evidence, Hypothesis, Prediction, Variant
from fitness_agents.mutation import AgentUncertaintySelector


def _variant(variant_id: str, code: str) -> Variant:
    return Variant(variant_id, code, code, code, sum(a != b for a, b in zip(code, "VDGV")), "pool")


def _prediction(variant_id: str, mean: float) -> Prediction:
    return Prediction(variant_id, mean, 0.1, (mean - 0.2, mean + 0.2), 0.1, {}, "model")


def test_agent_uq_default_is_invariant_to_fitness_prediction_means():
    observed = [_variant("wt", "VDGV"), _variant("seen", "FDGV")]
    candidates = [_variant("a", "FWGV"), _variant("b", "VDAL")]
    hypothesis = Hypothesis("h", "test", {39: ("F",), 40: ("W",)}, (), "+", "-")
    evidence = {
        "a": [Evidence("ea", "a", "kg", "support", 0.8, "kg", 0.9, 1)],
        "b": [Evidence("eb", "b", "kg", "weak", 0.1, "kg", 0.9, 1)],
    }
    selector = AgentUncertaintySelector(GenerationConfig(use_fitness_predictors=False))
    low_high = selector.score(
        candidates,
        observed_variants=observed,
        hypothesis=hypothesis,
        evidence=evidence,
        predictor_predictions=[[_prediction("a", -100), _prediction("b", 100)]],
    )
    high_low = selector.score(
        candidates,
        observed_variants=observed,
        hypothesis=hypothesis,
        evidence=evidence,
        predictor_predictions=[[_prediction("a", 100), _prediction("b", -100)]],
    )
    assert [item.utility for item in low_high] == pytest.approx(
        [item.utility for item in high_low]
    )
    assert all(item.predictor_score == 0.0 for item in low_high)


def test_optional_multi_predictor_interface_changes_only_when_enabled():
    candidates = [_variant("a", "FWGV"), _variant("b", "VDAL")]
    config = GenerationConfig(
        use_fitness_predictors=True,
        predictor_weight=1.0,
        hypothesis_weight=0.0,
        evidence_weight=0.0,
        prior_weight=0.0,
        uncertainty_beta=0.0,
    )
    selector = AgentUncertaintySelector(config)
    scores = selector.score(
        candidates,
        observed_variants=[_variant("wt", "VDGV")],
        hypothesis=None,
        evidence={},
        predictor_predictions=[
            [_prediction("a", 2.0), _prediction("b", 0.0)],
            [_prediction("a", 4.0), _prediction("b", 1.0)],
        ],
    )
    assert scores[0].utility > scores[1].utility
    assert scores[0].selection_driver == "kg_llm_uq_plus_predictor_ensemble"
    assert replace(config, use_fitness_predictors=False).use_fitness_predictors is False
