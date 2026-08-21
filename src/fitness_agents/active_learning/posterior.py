from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil
from typing import Any

import numpy as np

from fitness_agents.config import CalibratedPosteriorConfig, ModelConfig
from fitness_agents.contracts.schemas import FitnessObservation, Prediction, Variant

from .contracts import CalibratedPosteriorResult, PosteriorCalibrationSummary


def _aligned_predictions(
    predictions: Sequence[Prediction], variants: Sequence[Variant]
) -> list[Prediction]:
    by_id = {item.variant_id: item for item in predictions}
    missing = [item.variant_id for item in variants if item.variant_id not in by_id]
    if missing:
        raise ValueError(f"Posterior predictor omitted {len(missing)} variants")
    return [by_id[item.variant_id] for item in variants]


def _mixture_std(
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    center = mean_matrix @ weights
    second_moment = np.sum(
        weights[None, :] * (std_matrix**2 + mean_matrix**2), axis=1
    )
    return np.sqrt(np.maximum(second_moment - center**2, 0.0))


class VisibleHoldoutCalibratedPosterior:
    """Small visible-label-only ensemble with deterministic holdout calibration."""

    name = "visible_holdout_ensemble"
    min_visible_observations = 4

    def __init__(
        self,
        config: CalibratedPosteriorConfig,
        model_configs: Sequence[ModelConfig],
        predictor_factory: Callable[..., Any],
        *,
        seed: int,
    ) -> None:
        if not model_configs:
            raise ValueError("Active-learning posterior requires at least one predictor model")
        self.config = config
        self.model_configs = tuple(model_configs)
        self.predictor_factory = predictor_factory
        self.seed = seed
        self._predictors: tuple[Any, ...] = ()
        self._weights = np.full(len(self.model_configs), 1.0 / len(self.model_configs))
        self._bias = 0.0
        self._variance_scale = 1.0
        self._conformal_radius = 0.0
        self._summary: PosteriorCalibrationSummary | None = None
        self._staging_predictors: tuple[Any, ...] = ()

    @staticmethod
    def _align_visible(
        variants: Sequence[Variant], observations: Sequence[FitnessObservation]
    ) -> tuple[list[Variant], list[FitnessObservation]]:
        observation_by_id = {item.variant_id: item for item in observations}
        aligned_variants = [item for item in variants if item.variant_id in observation_by_id]
        if len(aligned_variants) != len(observation_by_id):
            missing = sorted(set(observation_by_id).difference(item.variant_id for item in variants))
            raise ValueError(f"Visible posterior observations lack {len(missing)} variants")
        aligned_observations = [observation_by_id[item.variant_id] for item in aligned_variants]
        return aligned_variants, aligned_observations

    def _fit_models(
        self,
        variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
        *,
        seed_offset: int,
    ) -> tuple[Any, ...]:
        fitted = []
        for index, model_config in enumerate(self.model_configs):
            predictor = self.predictor_factory(
                model_config, seed=self.seed + seed_offset + index * 1009
            )
            predictor.fit(variants, observations)
            fitted.append(predictor)
        return tuple(fitted)

    @staticmethod
    def _prediction_matrices(
        predictors: Sequence[Any], variants: Sequence[Variant]
    ) -> tuple[np.ndarray, np.ndarray, list[list[Prediction]]]:
        prediction_sets = [
            _aligned_predictions(predictor.predict(variants), variants)
            for predictor in predictors
        ]
        means = np.column_stack(
            [[item.fitness_mean for item in predictions] for predictions in prediction_sets]
        )
        stds = np.column_stack(
            [
                [max(item.fitness_std, 1e-12) for item in predictions]
                for predictions in prediction_sets
            ]
        )
        return means, stds, prediction_sets

    def _fit_calibration(
        self,
        variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
    ) -> tuple[int, int, str]:
        count = len(variants)
        required = self.config.min_training_size + self.config.min_calibration_size
        if count < required:
            return count, 0, "insufficient_visible_data"

        calibration_size = max(
            self.config.min_calibration_size,
            ceil(count * self.config.calibration_fraction),
        )
        calibration_size = min(calibration_size, count - self.config.min_training_size)
        rng = np.random.default_rng(self.seed + count * 7919)
        shuffled = rng.permutation(count)
        calibration_indices = {int(item) for item in shuffled[-calibration_size:]}
        train_variants = [item for index, item in enumerate(variants) if index not in calibration_indices]
        train_observations = [
            item for index, item in enumerate(observations) if index not in calibration_indices
        ]
        calibration_variants = [
            item for index, item in enumerate(variants) if index in calibration_indices
        ]
        calibration_observations = [
            item for index, item in enumerate(observations) if index in calibration_indices
        ]
        staging = self._fit_models(
            train_variants, train_observations, seed_offset=calibration_size * 17
        )
        self._staging_predictors = staging
        means, stds, _ = self._prediction_matrices(staging, calibration_variants)
        targets = np.asarray([item.fitness for item in calibration_observations], dtype=float)

        if means.shape[1] == 1:
            weights = np.ones(1, dtype=float)
        else:
            solution = np.linalg.lstsq(
                np.column_stack([means, np.ones(len(means))]), targets, rcond=None
            )[0]
            weights = np.maximum(solution[:-1], 0.0)
            if float(weights.sum()) <= 1e-12:
                weights = np.ones(means.shape[1], dtype=float)
            weights /= weights.sum()
        bias = float(np.mean(targets - means @ weights))
        calibrated_mean = means @ weights + bias
        raw_std = _mixture_std(means, stds, weights)
        residuals = np.abs(targets - calibrated_mean)
        denominator = float(np.sqrt(np.mean(np.maximum(raw_std, self.config.min_std) ** 2)))
        scale = float(np.sqrt(np.mean(residuals**2)) / max(denominator, self.config.min_std))
        lower, upper = self.config.variance_scale_bounds
        self._weights = weights
        self._bias = bias
        self._variance_scale = float(np.clip(scale, lower, upper))
        quantile = min(
            1.0,
            ceil((len(residuals) + 1) * (1 - self.config.conformal_alpha))
            / len(residuals),
        )
        self._conformal_radius = float(
            np.quantile(residuals, quantile, method="higher")
        )
        return len(train_variants), len(calibration_variants), "calibrated"

    def fit(
        self,
        variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
    ) -> VisibleHoldoutCalibratedPosterior:
        self._weights = np.full(len(self.model_configs), 1.0 / len(self.model_configs))
        self._bias = 0.0
        self._variance_scale = 1.0
        self._conformal_radius = 0.0
        self._staging_predictors = ()
        aligned_variants, aligned_observations = self._align_visible(variants, observations)
        if len(aligned_variants) < self.min_visible_observations:
            raise ValueError(
                "Active-learning posterior requires at least "
                f"{self.min_visible_observations} visible observations"
            )
        train_count, calibration_count, status = self._fit_calibration(
            aligned_variants, aligned_observations
        )
        final_variants = aligned_variants
        final_observations = aligned_observations
        if self.config.refit_full or not self._staging_predictors:
            self._predictors = self._fit_models(
                final_variants,
                final_observations,
                seed_offset=0,
            )
        else:
            self._predictors = self._staging_predictors
        versions = tuple(str(item.model_version) for item in self._predictors)
        self._summary = PosteriorCalibrationSummary(
            plugin=self.name,
            status=status,
            visible_observations=len(aligned_variants),
            training_observations=train_count,
            calibration_observations=calibration_count,
            model_versions=versions,
            model_weights=tuple(float(item) for item in self._weights),
            bias=self._bias,
            variance_scale=self._variance_scale,
            conformal_radius=self._conformal_radius,
            conformal_alpha=self.config.conformal_alpha,
            refit_full=self.config.refit_full,
        )
        return self

    def predict(self, variants: Sequence[Variant]) -> CalibratedPosteriorResult:
        if self._summary is None or not self._predictors:
            raise RuntimeError("Active-learning posterior must be fitted before prediction")
        if not variants:
            return CalibratedPosteriorResult((), self._summary)
        means, stds, prediction_sets = self._prediction_matrices(self._predictors, variants)
        posterior_mean = means @ self._weights + self._bias
        posterior_std = np.maximum(
            _mixture_std(means, stds, self._weights) * self._variance_scale,
            self.config.min_std,
        )
        output = []
        version = (
            f"active-learning:{self.name}:n{self._summary.visible_observations}:"
            f"seed{self.seed}"
        )
        for index, variant in enumerate(variants):
            radius = max(
                self._conformal_radius,
                1.645 * float(posterior_std[index]),
            )
            component_scores = {
                f"model_{model_index}_mean": float(means[index, model_index])
                for model_index in range(means.shape[1])
            }
            component_scores.update(
                {
                    "calibration_bias": self._bias,
                    "variance_scale": self._variance_scale,
                    "conformal_radius": self._conformal_radius,
                }
            )
            output.append(
                Prediction(
                    variant_id=variant.variant_id,
                    fitness_mean=float(posterior_mean[index]),
                    fitness_std=float(posterior_std[index]),
                    interval_90=(
                        float(posterior_mean[index] - radius),
                        float(posterior_mean[index] + radius),
                    ),
                    ood_score=max(
                        predictions[index].ood_score for predictions in prediction_sets
                    ),
                    component_scores=component_scores,
                    model_version=version,
                )
            )
        return CalibratedPosteriorResult(tuple(output), self._summary)
