from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from fitness_agents.config import ActiveLearningConfig, ModelConfig
from fitness_agents.contracts.schemas import FitnessObservation, Variant

from .acquisition import HybridBatchAcquisition
from .contracts import (
    CalibratedPosteriorResult,
    HybridBatchSelection,
    HybridScoreResult,
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
        return self.posterior.fit(observed_variants, observations).predict(candidates)

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

