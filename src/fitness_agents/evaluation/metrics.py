from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, ndcg_score

from fitness_agents.contracts.schemas import FitnessObservation, Prediction

SUPPORTED_PREDICTION_METRICS = frozenset(
    {
        "spearman",
        "pearson",
        "mse",
        "rmse",
        "ndcg",
        "top_k_hit",
        "top_k_recall",
        "regret_at_k",
        "interval_90_coverage",
        "gaussian_nll",
    }
)


def _safe_correlation(kind: str, truth: np.ndarray, predicted: np.ndarray) -> float:
    if len(truth) < 2 or np.all(truth == truth[0]) or np.all(predicted == predicted[0]):
        return 0.0
    result = spearmanr(truth, predicted) if kind == "spearman" else pearsonr(truth, predicted)
    return float(result.statistic)


def prediction_metrics(
    predictions: Sequence[Prediction],
    observations: Sequence[FitnessObservation],
    *,
    metrics: Sequence[str] | None = None,
    top_k: int = 10,
) -> dict[str, float]:
    selected_metrics = tuple(metrics or sorted(SUPPORTED_PREDICTION_METRICS))
    unknown = set(selected_metrics).difference(SUPPORTED_PREDICTION_METRICS)
    if unknown:
        raise ValueError(f"Unsupported prediction metrics: {sorted(unknown)}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    prediction_map = {prediction.variant_id: prediction for prediction in predictions}
    aligned = [observation for observation in observations if observation.variant_id in prediction_map]
    if not aligned:
        return {}
    truth = np.asarray([observation.fitness for observation in aligned], dtype=float)
    mean = np.asarray([prediction_map[item.variant_id].fitness_mean for item in aligned], dtype=float)
    std = np.asarray([max(prediction_map[item.variant_id].fitness_std, 1e-8) for item in aligned])
    intervals = [prediction_map[item.variant_id].interval_90 for item in aligned]
    coverage = np.mean([low <= target <= high for target, (low, high) in zip(truth, intervals)])
    gaussian_nll = np.mean(np.log(std) + 0.5 * ((truth - mean) / std) ** 2)
    k = min(top_k, len(aligned))
    ids = np.asarray([item.variant_id for item in aligned])
    truth_order = np.lexsort((ids, -truth))[:k]
    prediction_order = np.lexsort((ids, -mean))[:k]
    true_top = set(ids[truth_order])
    predicted_top = set(ids[prediction_order])
    intersection = true_top.intersection(predicted_top)
    values: dict[str, float] = {}
    if "spearman" in selected_metrics:
        values["spearman"] = _safe_correlation("spearman", truth, mean)
    if "pearson" in selected_metrics:
        values["pearson"] = _safe_correlation("pearson", truth, mean)
    if "mse" in selected_metrics or "rmse" in selected_metrics:
        mse = float(mean_squared_error(truth, mean))
        values["mse"] = mse
        values["rmse"] = float(np.sqrt(mse))
    if "ndcg" in selected_metrics:
        relevance = truth - truth.min()
        if len(aligned) == 1:
            values["ndcg"] = 1.0
        elif np.all(relevance == 0):
            values["ndcg"] = 0.0
        else:
            values["ndcg"] = float(
                ndcg_score(relevance.reshape(1, -1), mean.reshape(1, -1))
            )
    if "top_k_hit" in selected_metrics:
        values["top_k_hit"] = float(bool(intersection))
    if "top_k_recall" in selected_metrics:
        values["top_k_recall"] = float(len(intersection) / max(len(true_top), 1))
    if "regret_at_k" in selected_metrics:
        values["regret_at_k"] = float(truth.max() - truth[prediction_order].max())
    if "interval_90_coverage" in selected_metrics:
        values["interval_90_coverage"] = float(coverage)
    if "gaussian_nll" in selected_metrics:
        values["gaussian_nll"] = float(gaussian_nll)
    return {"n": float(len(aligned)), **{key: values[key] for key in selected_metrics}}


def loop_round_metrics(
    all_visible: Sequence[FitnessObservation],
    newly_revealed: Sequence[FitnessObservation],
    *,
    total_pool_size: int,
    selected_model_ranks: Sequence[int],
) -> dict[str, float]:
    visible = np.asarray([item.fitness for item in all_visible], dtype=float)
    batch = np.asarray([item.fitness for item in newly_revealed], dtype=float)
    ranks = np.asarray(list(selected_model_ranks), dtype=float)
    return {
        "best_seen_fitness": float(visible.max()),
        "visible_mean_fitness": float(visible.mean()),
        "batch_best_fitness": float(batch.max()),
        "batch_mean_fitness": float(batch.mean()),
        "batch_median_fitness": float(np.median(batch)),
        "mean_selected_model_rank": float(ranks.mean()) if len(ranks) else float("nan"),
        "mean_selected_model_rank_fraction": (
            float(ranks.mean() / max(total_pool_size, 1)) if len(ranks) else float("nan")
        ),
    }
