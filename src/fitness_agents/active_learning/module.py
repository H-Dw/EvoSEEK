from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from fitness_agents.config import ActiveLearningConfig, ModelConfig
from fitness_agents.contracts.schemas import FitnessObservation, Prediction, Variant

from .acquisition import HybridBatchAcquisition
from .contracts import (
    CalibratedPosteriorResult,
    HybridBatchSelection,
    HybridScoreResult,
    PosteriorCalibrationSummary,
)
from .posterior import VisibleHoldoutCalibratedPosterior


class LightweightCalibratedHybridModule:
    """Pluggable composition of a calibrated posterior and hybrid acquisition."""

    name = "lightweight_calibrated_hybrid"

    def __init__(
        self,
        config: ActiveLearningConfig,
        *,
        fallback_model: ModelConfig,
        predictor_factory: Callable[..., Any],
        seed: int,
    ) -> None:
        self.config = config
        self.seed = seed
        models = config.posterior.predictor_models or (fallback_model,)
        if config.posterior.plugin != VisibleHoldoutCalibratedPosterior.name:
            raise ValueError(
                f"Unknown posterior plugin {config.posterior.plugin!r}; "
                f"available={[VisibleHoldoutCalibratedPosterior.name]}"
            )
        if config.acquisition.plugin != HybridBatchAcquisition.name:
            raise ValueError(
                f"Unknown hybrid acquisition plugin {config.acquisition.plugin!r}; "
                f"available={[HybridBatchAcquisition.name]}"
            )
        self.posterior = VisibleHoldoutCalibratedPosterior(
            config.posterior,
            models,
            predictor_factory,
            seed=seed,
        )
        self.acquisition = HybridBatchAcquisition(config.acquisition)

    def fit_predict(
        self,
        observed_variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
        candidates: Sequence[Variant],
    ) -> CalibratedPosteriorResult:
        if len(observations) < VisibleHoldoutCalibratedPosterior.min_visible_observations:
            # Cold-start warmup: too few visible observations to fit even an
            # uncalibrated ensemble, so fall back to a neutral constant
            # posterior and let knowledge/exploration terms drive acquisition.
            return self._warmup_posterior(observations, candidates)
        return self.posterior.fit(observed_variants, observations).predict(candidates)

    def _warmup_posterior(
        self,
        observations: Sequence[FitnessObservation],
        candidates: Sequence[Variant],
    ) -> CalibratedPosteriorResult:
        center = (
            sum(item.fitness for item in observations) / len(observations)
            if observations
            else 0.0
        )
        std = 1.0
        version = (
            f"active-learning:warmup:n{len(observations)}:seed{self.seed}"
        )
        predictions = tuple(
            Prediction(
                variant_id=variant.variant_id,
                fitness_mean=center,
                fitness_std=std,
                interval_90=(center - 1.645 * std, center + 1.645 * std),
                ood_score=1.0,
                component_scores={"warmup_constant": center},
                model_version=version,
            )
            for variant in candidates
        )
        summary = PosteriorCalibrationSummary(
            plugin=VisibleHoldoutCalibratedPosterior.name,
            status="warmup_insufficient_data",
            visible_observations=len(observations),
            training_observations=0,
            calibration_observations=0,
            model_versions=(),
            model_weights=(),
            bias=0.0,
            variance_scale=1.0,
            conformal_radius=0.0,
            conformal_alpha=self.config.posterior.conformal_alpha,
            refit_full=self.config.posterior.refit_full,
        )
        return CalibratedPosteriorResult(predictions, summary)

    def score(
        self,
        posterior: CalibratedPosteriorResult,
        knowledge_scores: dict[str, float],
    ) -> HybridScoreResult:
        return self.acquisition.score(posterior.predictions, knowledge_scores)

    def select(
        self,
        variants: Sequence[Variant],
        scores: HybridScoreResult,
        budget: int,
        *,
        knowledge_scores: dict[str, float],
    ) -> HybridBatchSelection:
        return self.acquisition.select(
            variants,
            scores,
            budget,
            knowledge_scores=knowledge_scores,
        )

