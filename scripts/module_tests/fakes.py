from __future__ import annotations

from typing import Any

from common import fitness_for


class DeterministicExternalBackend:
    """A checkpoint-free backend used only to validate the external predictor contract."""

    model_version = "deterministic-external-v1"

    def __init__(self, context: Any) -> None:
        self.context = context
        self._fitted = False
        self._offset = 0.0

    def fit(
        self,
        variants,
        observations,
        validation_variants=None,
        validation_observations=None,
    ) -> None:
        del variants, validation_variants, validation_observations
        if not observations:
            raise ValueError("The fake external backend requires observations")
        self._offset = sum(item.fitness for item in observations) / len(observations) * 0.01
        self._fitted = True

    def predict(self, variants):
        if not self._fitted:
            raise RuntimeError("fit must be called before predict")
        # Reverse output deliberately. ExternalPredictorAdapter must restore request order.
        return [
            {
                "variant_id": item.variant_id,
                "fitness_mean": fitness_for(item.variant) + self._offset,
                "fitness_std": 0.15,
                "interval_90": (
                    fitness_for(item.variant) + self._offset - 0.25,
                    fitness_for(item.variant) + self._offset + 0.25,
                ),
                "ood_score": item.mutation_count / 4.0,
                "component_scores": {"fake_external": fitness_for(item.variant)},
            }
            for item in reversed(list(variants))
        ]


def create_external_backend(context: Any) -> DeterministicExternalBackend:
    return DeterministicExternalBackend(context)

