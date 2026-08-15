from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from fitness_agents.contracts.schemas import Prediction, Variant
from fitness_agents.features.gb1 import hamming_distance


class _BasePolicy:
    name = "base"

    def __init__(self, *, knowledge_weight: float = 0.0, beta: float = 1.5) -> None:
        self.knowledge_weight = knowledge_weight
        self.beta = beta

    def _base(self, prediction: Prediction, rng: np.random.Generator) -> float:
        raise NotImplementedError

    def score(
        self,
        predictions: Sequence[Prediction],
        knowledge_scores: dict[str, float],
        rng: np.random.Generator,
    ) -> dict[str, float]:
        return {
            prediction.variant_id: self._base(prediction, rng)
            + self.knowledge_weight * knowledge_scores.get(prediction.variant_id, 0.0)
            for prediction in predictions
        }

    def select(
        self,
        variants: Sequence[Variant],
        predictions: Sequence[Prediction],
        scores: dict[str, float],
        budget: int,
        diversity_lambda: float,
    ) -> list[str]:
        by_id = {variant.variant_id: variant for variant in variants}
        available = set(by_id).intersection(scores)
        selected: list[str] = []
        while available and len(selected) < budget:
            def adjusted(variant_id: str) -> tuple[float, str]:
                if not selected or diversity_lambda <= 0:
                    penalty = 0.0
                else:
                    nearest_similarity = max(
                        1.0 - hamming_distance(
                            by_id[variant_id].variant, by_id[chosen].variant
                        ) / 4.0
                        for chosen in selected
                    )
                    penalty = diversity_lambda * nearest_similarity
                return scores[variant_id] - penalty, variant_id

            choice = max(available, key=adjusted)
            selected.append(choice)
            available.remove(choice)
        return selected


class RandomPolicy(_BasePolicy):
    name = "random"

    def _base(self, prediction: Prediction, rng: np.random.Generator) -> float:
        return float(rng.random())


class GreedyPolicy(_BasePolicy):
    name = "greedy"

    def _base(self, prediction: Prediction, rng: np.random.Generator) -> float:
        return prediction.fitness_mean


class UCBPolicy(_BasePolicy):
    name = "ucb"

    def _base(self, prediction: Prediction, rng: np.random.Generator) -> float:
        return prediction.fitness_mean + self.beta * prediction.fitness_std


class ThompsonPolicy(_BasePolicy):
    name = "thompson"

    def _base(self, prediction: Prediction, rng: np.random.Generator) -> float:
        return float(rng.normal(prediction.fitness_mean, max(prediction.fitness_std, 1e-8)))


POLICIES: dict[str, Callable[..., _BasePolicy]] = {
    "random": RandomPolicy,
    "greedy": GreedyPolicy,
    "ucb": UCBPolicy,
    "thompson": ThompsonPolicy,
    "ts": ThompsonPolicy,
}


def register_policy(name: str, factory: Callable[..., _BasePolicy]) -> None:
    if not name or name in POLICIES:
        raise ValueError(f"Policy name must be new and non-empty: {name!r}")
    POLICIES[name] = factory


def create_policy(name: str, *, beta: float, knowledge_weight: float):
    try:
        return POLICIES[name](beta=beta, knowledge_weight=knowledge_weight)
    except KeyError as error:
        raise ValueError(f"Unknown acquisition policy {name!r}; available={sorted(POLICIES)}") from error
