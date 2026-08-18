from __future__ import annotations

from collections.abc import Sequence
from math import floor

from fitness_agents.config import HybridBatchAcquisitionConfig
from fitness_agents.contracts.schemas import Prediction, Variant

from .contracts import HybridBatchSelection, HybridCandidateScore, HybridScoreResult

ARMS = ("exploitation", "exploration", "knowledge")


def _rank_scale(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    unique = sorted(set(values.values()))
    if len(unique) == 1:
        return {key: 0.5 for key in values}
    rank = {value: index / (len(unique) - 1) for index, value in enumerate(unique)}
    return {key: float(rank[value]) for key, value in values.items()}


def _sequence_distance(left: str, right: str) -> float:
    length = max(len(left), len(right), 1)
    mismatches = sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))
    return mismatches / length


def _integer_quotas(budget: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {name: budget * fractions[name] for name in ARMS}
    quotas = {name: floor(raw[name]) for name in ARMS}
    remaining = budget - sum(quotas.values())
    order = sorted(ARMS, key=lambda name: (raw[name] - quotas[name], -ARMS.index(name)), reverse=True)
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


class HybridBatchAcquisition:
    """Quota-based hybrid of exploitation, epistemic exploration, and knowledge."""

    name = "hybrid_batch"

    def __init__(self, config: HybridBatchAcquisitionConfig) -> None:
        self.config = config

    def _effective_fractions(self, knowledge_available: bool) -> dict[str, float]:
        fractions = {
            "exploitation": self.config.exploitation_fraction,
            "exploration": self.config.exploration_fraction,
            "knowledge": self.config.knowledge_fraction if knowledge_available else 0.0,
        }
        if knowledge_available:
            return fractions
        normalizer = fractions["exploitation"] + fractions["exploration"]
        if normalizer <= 0:
            fractions["exploitation"] = 1.0
        else:
            fractions["exploitation"] /= normalizer
            fractions["exploration"] /= normalizer
        return fractions

    def score(
        self,
        predictions: Sequence[Prediction],
        knowledge_scores: dict[str, float],
    ) -> HybridScoreResult:
        exploitation_raw = {
            item.variant_id: item.fitness_mean
            + self.config.ucb_beta * item.fitness_std
            - self.config.ood_penalty * item.ood_score
            for item in predictions
        }
        exploration_raw = {
            item.variant_id: item.fitness_std
            - self.config.ood_penalty * item.ood_score
            for item in predictions
        }
        knowledge_raw = {
            item.variant_id: knowledge_scores.get(item.variant_id, 0.0)
            + self.config.knowledge_fitness_weight * item.fitness_mean
            - self.config.ood_penalty * item.ood_score
            for item in predictions
        }
        exploitation = _rank_scale(exploitation_raw)
        exploration = _rank_scale(exploration_raw)
        knowledge = _rank_scale(knowledge_raw)
        fractions = self._effective_fractions(
            any(abs(value) > 1e-12 for value in knowledge_scores.values())
        )
        by_id = {item.variant_id: item for item in predictions}
        scores = []
        for variant_id in sorted(by_id):
            prediction = by_id[variant_id]
            composite = (
                fractions["exploitation"] * exploitation[variant_id]
                + fractions["exploration"] * exploration[variant_id]
                + fractions["knowledge"] * knowledge[variant_id]
            )
            scores.append(
                HybridCandidateScore(
                    variant_id=variant_id,
                    exploitation=exploitation[variant_id],
                    exploration=exploration[variant_id],
                    knowledge=knowledge[variant_id],
                    composite=float(composite),
                    fitness_mean=prediction.fitness_mean,
                    fitness_std=prediction.fitness_std,
                    ood_score=prediction.ood_score,
                    knowledge_score=float(knowledge_scores.get(variant_id, 0.0)),
                )
            )
        return HybridScoreResult(tuple(scores))

    def select(
        self,
        variants: Sequence[Variant],
        score_result: HybridScoreResult,
        budget: int,
        *,
        knowledge_scores: dict[str, float],
    ) -> HybridBatchSelection:
        if budget < 0:
            raise ValueError("Hybrid acquisition budget must be non-negative")
        by_id = {item.variant_id: item for item in variants}
        score_by_id = score_result.by_id()
        available = set(by_id).intersection(score_by_id)
        target = min(budget, len(available))
        knowledge_available = any(
            abs(knowledge_scores.get(variant_id, 0.0)) > 1e-12 for variant_id in available
        )
        fractions = self._effective_fractions(knowledge_available)
        quotas = _integer_quotas(target, fractions)
        selected: list[str] = []
        selected_by_arm: dict[str, list[str]] = {name: [] for name in ARMS}

        def arm_value(variant_id: str, arm: str) -> tuple[float, str]:
            score = score_by_id[variant_id]
            base = float(getattr(score, arm))
            if not selected or self.config.diversity_lambda <= 0:
                penalty = 0.0
            else:
                nearest_similarity = max(
                    1.0
                    - _sequence_distance(
                        by_id[variant_id].variant,
                        by_id[chosen].variant,
                    )
                    for chosen in selected
                )
                penalty = self.config.diversity_lambda * nearest_similarity
            return base - penalty, variant_id

        while available and len(selected) < target:
            progressed = False
            for arm in ARMS:
                if len(selected_by_arm[arm]) >= quotas[arm] or not available:
                    continue
                choice = max(available, key=lambda variant_id, current=arm: arm_value(variant_id, current))
                selected.append(choice)
                selected_by_arm[arm].append(choice)
                available.remove(choice)
                progressed = True
            if not progressed:
                break
        if available and len(selected) < target:
            fill = sorted(
                available,
                key=lambda variant_id: (score_by_id[variant_id].composite, variant_id),
                reverse=True,
            )
            selected.extend(fill[: target - len(selected)])
        return HybridBatchSelection(
            plugin=self.name,
            selected_ids=tuple(selected),
            quotas=quotas,
            selected_by_arm={key: tuple(value) for key, value in selected_by_arm.items()},
            effective_fractions={key: float(value) for key, value in fractions.items()},
            knowledge_available=knowledge_available,
        )
