from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, ndcg_score

from fitness_agents.contracts.schemas import FitnessObservation, Prediction


def _safe_correlation(kind: str, truth: np.ndarray, predicted: np.ndarray) -> float:
    if len(truth) < 2 or np.all(truth == truth[0]) or np.all(predicted == predicted[0]):
        return 0.0
    result = spearmanr(truth, predicted) if kind == "spearman" else pearsonr(truth, predicted)
    return float(result.statistic)


def prediction_metrics(
    predictions: Sequence[Prediction], observations: Sequence[FitnessObservation]
) -> dict[str, float]:
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
    return {
        "n": float(len(aligned)),
        "spearman": _safe_correlation("spearman", truth, mean),
        "pearson": _safe_correlation("pearson", truth, mean),
        "rmse": float(np.sqrt(mean_squared_error(truth, mean))),
        "ndcg": float(ndcg_score(truth.reshape(1, -1), mean.reshape(1, -1))),
        "interval_90_coverage": float(coverage),
        "gaussian_nll": float(gaussian_nll),
    }


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

