import pytest

from fitness_agents.contracts.schemas import FitnessObservation, Prediction
from fitness_agents.evaluation.metrics import prediction_metrics


def _pred(variant_id: str, mean: float) -> Prediction:
    return Prediction(variant_id, mean, 0.2, (mean - 0.4, mean + 0.4), 0.0, {}, "m")


def test_configurable_top_k_mse_and_regret_metrics():
    observations = [
        FitnessObservation("a", 3.0, "test", 0),
        FitnessObservation("b", 2.0, "test", 0),
        FitnessObservation("c", 1.0, "test", 0),
    ]
    perfect = prediction_metrics(
        [_pred("a", 3.0), _pred("b", 2.0), _pred("c", 1.0)],
        observations,
        metrics=("mse", "top_k_hit", "top_k_recall", "regret_at_k"),
        top_k=2,
    )
    assert perfect == {
        "n": 3.0,
        "mse": 0.0,
        "top_k_hit": 1.0,
        "top_k_recall": 1.0,
        "regret_at_k": 0.0,
    }
    reverse = prediction_metrics(
        [_pred("a", 1.0), _pred("b", 2.0), _pred("c", 3.0)],
        observations,
        metrics=("top_k_hit", "top_k_recall", "regret_at_k"),
        top_k=1,
    )
    assert reverse["top_k_hit"] == 0.0
    assert reverse["top_k_recall"] == 0.0
    assert reverse["regret_at_k"] == 2.0


def test_metric_selection_rejects_unknown_names_and_caps_k():
    observations = [FitnessObservation("a", 1.0, "test", 0)]
    metrics = prediction_metrics(
        [_pred("a", 1.0)], observations, metrics=("top_k_recall",), top_k=10
    )
    assert metrics["top_k_recall"] == 1.0
    with pytest.raises(ValueError, match="Unsupported"):
        prediction_metrics([_pred("a", 1.0)], observations, metrics=("magic",))


def test_ndcg_accepts_negative_fitness_values():
    observations = [
        FitnessObservation("a", -3.0, "test", 0),
        FitnessObservation("b", -2.0, "test", 0),
        FitnessObservation("c", -1.0, "test", 0),
    ]

    metrics = prediction_metrics(
        [_pred("a", 0.0), _pred("b", 1.0), _pred("c", 2.0)],
        observations,
        metrics=("ndcg",),
        top_k=2,
    )

    assert metrics["ndcg"] == pytest.approx(1.0)
