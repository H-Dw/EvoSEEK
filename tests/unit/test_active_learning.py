from __future__ import annotations

from dataclasses import replace
from itertools import islice, product

import pytest

from fitness_agents.active_learning import (
    HybridBatchAcquisition,
    VisibleHoldoutCalibratedPosterior,
)
from fitness_agents.active_learning.module import LightweightCalibratedHybridModule
from fitness_agents.config import (
    ActiveLearningConfig,
    CalibratedPosteriorConfig,
    GenerationConfig,
    HybridBatchAcquisitionConfig,
    ModelConfig,
    load_experiment_config,
)
from fitness_agents.contracts.schemas import FitnessObservation, Prediction, Variant
from fitness_agents.models import create_predictor


def _variant(identifier: str, code: str, role: str = "initial_observed") -> Variant:
    return Variant(
        identifier,
        code,
        code,
        code,
        sum(left != right for left, right in zip(code, "VDGV", strict=True)),
        role,
    )


def _prediction(identifier: str, mean: float, std: float, ood: float = 0.0) -> Prediction:
    return Prediction(
        identifier,
        mean,
        std,
        (mean - 1.645 * std, mean + 1.645 * std),
        ood,
        {},
        "test",
    )


def test_visible_holdout_posterior_calibrates_and_returns_structured_predictions():
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    codes = ["".join(item) for item in islice(product(alphabet, repeat=4), 32)]
    variants = [_variant(f"v{index}", code) for index, code in enumerate(codes)]
    observations = [
        FitnessObservation(
            item.variant_id,
            0.4 * item.mutation_count + 0.03 * index,
            "initial_observed",
            0,
            "test",
        )
        for index, item in enumerate(variants[:24])
    ]
    model = ModelConfig(ridge_members=2, extra_trees_estimators=12, bootstrap_fraction=0.8)
    alternate_model = replace(model, ridge_alpha=1.0)
    config = CalibratedPosteriorConfig(
        predictor_models=(model, alternate_model),
        calibration_fraction=0.25,
        min_calibration_size=4,
        min_training_size=8,
    )
    posterior = VisibleHoldoutCalibratedPosterior(
        config,
        (model, alternate_model),
        create_predictor,
        seed=17,
    )

    result = posterior.fit(variants[:24], observations).predict(variants[24:])

    assert result.calibration.status == "calibrated"
    assert result.calibration.visible_observations == 24
    assert result.calibration.training_observations + result.calibration.calibration_observations == 24
    assert sum(result.calibration.model_weights) == pytest.approx(1.0)
    assert len(result.calibration.model_weights) == 2
    assert all(item >= 0 for item in result.calibration.model_weights)
    assert len(result.predictions) == 8
    assert all(item.fitness_std > 0 for item in result.predictions)
    assert all(item.interval_90[0] <= item.fitness_mean <= item.interval_90[1] for item in result.predictions)
    assert all(item.model_version.startswith("active-learning:visible_holdout_ensemble") for item in result.predictions)


def test_hybrid_batch_uses_explicit_arm_quotas_without_duplicate_variants():
    variants = [
        _variant("exploit-a", "AAAA", "oracle_pool"),
        _variant("exploit-b", "AAAC", "oracle_pool"),
        _variant("explore", "CCCC", "oracle_pool"),
        _variant("knowledge", "DDDD", "oracle_pool"),
        _variant("other-a", "EEEE", "oracle_pool"),
        _variant("other-b", "FFFF", "oracle_pool"),
    ]
    predictions = [
        _prediction("exploit-a", 2.0, 0.1),
        _prediction("exploit-b", 1.8, 0.1),
        _prediction("explore", 0.2, 1.5),
        _prediction("knowledge", 0.5, 0.2),
        _prediction("other-a", 0.4, 0.3),
        _prediction("other-b", 0.3, 0.4),
    ]
    knowledge = {"knowledge": 10.0}
    acquisition = HybridBatchAcquisition(
        HybridBatchAcquisitionConfig(ucb_beta=0.0, diversity_lambda=0.0)
    )
    scores = acquisition.score(predictions, knowledge)

    selection = acquisition.select(variants, scores, 4, knowledge_scores=knowledge)

    assert selection.quotas == {"exploitation": 2, "exploration": 1, "knowledge": 1}
    assert selection.selected_by_arm["exploitation"] == ("exploit-a", "exploit-b")
    assert selection.selected_by_arm["exploration"] == ("explore",)
    assert selection.selected_by_arm["knowledge"] == ("knowledge",)
    assert len(selection.selected_ids) == len(set(selection.selected_ids)) == 4

    no_knowledge_scores = acquisition.score(predictions, {})
    no_knowledge = acquisition.select(
        variants,
        no_knowledge_scores,
        4,
        knowledge_scores={},
    )
    assert no_knowledge.quotas["knowledge"] == 0
    assert sum(no_knowledge.quotas.values()) == 4


def test_active_learning_config_is_explicit_and_sample_config_loads(experiment_config):
    loaded = load_experiment_config(
        "configs/experiments/knowledge_agent_active_learning.yaml"
    )
    assert loaded.generation.selection_driver == "active_learning"
    assert loaded.active_learning.enabled is True
    assert loaded.active_learning.module == "lightweight_calibrated_hybrid"
    assert loaded.active_learning.posterior.predictor_models[0].name == (
        "onehot_heterogeneous_ensemble"
    )
    assert loaded.active_learning.acquisition.plugin == "hybrid_batch"

    with pytest.raises(ValueError, match="must be configured together"):
        replace(
            experiment_config,
            generation=GenerationConfig(selection_driver="active_learning"),
        )


def test_active_learning_module_warmup_when_visible_data_below_minimum():
    model = ModelConfig(ridge_members=2, extra_trees_estimators=12, bootstrap_fraction=0.8)
    config = ActiveLearningConfig(
        enabled=True,
        posterior=CalibratedPosteriorConfig(
            predictor_models=(model,),
            calibration_fraction=0.25,
            min_calibration_size=4,
            min_training_size=8,
        ),
        acquisition=HybridBatchAcquisitionConfig(
            exploitation_fraction=0.50,
            exploration_fraction=0.25,
            knowledge_fraction=0.25,
            ucb_beta=1.0,
            diversity_lambda=0.10,
        ),
    )
    module = LightweightCalibratedHybridModule(
        config,
        fallback_model=model,
        predictor_factory=create_predictor,
        seed=17,
    )
    wild_type = _variant("wt", "VDGV")
    observations = [
        FitnessObservation("wt", 1.0, "initial_observed", 0, "fold_initial_observed")
    ]
    candidates = [
        _variant("cand-a", "ADGV", "oracle_pool"),
        _variant("cand-b", "VDGA", "oracle_pool"),
        _variant("cand-c", "AAAA", "oracle_pool"),
        _variant("cand-d", "FFFF", "oracle_pool"),
    ]

    result = module.fit_predict([wild_type], observations, candidates)

    assert result.calibration.status == "warmup_insufficient_data"
    assert result.calibration.visible_observations == 1
    assert len(result.predictions) == 4
    assert all(item.fitness_mean == 1.0 for item in result.predictions)
    assert all(item.fitness_std > 0 for item in result.predictions)
    # Budget 3 splits 1/1/1 across arms, so the knowledge arm must claim cand-b.
    knowledge = {"cand-b": 5.0}
    scores = module.score(result, knowledge)
    selection = module.select(candidates, scores, 3, knowledge_scores=knowledge)
    assert selection.quotas == {"exploitation": 1, "exploration": 1, "knowledge": 1}
    assert selection.selected_by_arm["knowledge"] == ("cand-b",)
    assert len(selection.selected_ids) == 3
