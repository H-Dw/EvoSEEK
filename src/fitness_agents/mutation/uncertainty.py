from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from fitness_agents.config import GenerationConfig
from fitness_agents.contracts.schemas import (
    DesignScore,
    Evidence,
    Hypothesis,
    Prediction,
    Variant,
)


def _hypothesis_score(
    variant: Variant,
    hypothesis: Hypothesis | None,
    position_to_index: Mapping[int, int],
) -> float:
    if hypothesis is None or not hypothesis.preferred_residues:
        return 0.0
    matches = 0
    tested = 0
    for position, residues in hypothesis.preferred_residues.items():
        if position not in position_to_index or not residues:
            continue
        tested += 1
        matches += variant.variant[position_to_index[position]] in residues
    return float(matches / tested) if tested else 0.0


def _evidence_score(items: Sequence[Evidence]) -> float:
    if not items:
        return 0.0
    selectable = [item for item in items if item.contributes_to_selection]
    if not selectable:
        return 0.0
    weights = np.asarray([max(float(item.confidence), 1e-8) for item in selectable])
    values = np.asarray([float(item.score) for item in selectable])
    return float(np.average(values, weights=weights))


def _predictor_ensemble_scores(
    candidates: Sequence[Variant],
    prediction_sets: Sequence[Sequence[Prediction]],
) -> dict[str, float]:
    """Average within-model z-scores so one predictor cannot dominate by output scale."""

    if not prediction_sets:
        return {item.variant_id: 0.0 for item in candidates}
    accum: dict[str, list[float]] = {item.variant_id: [] for item in candidates}
    for predictions in prediction_sets:
        mapping = {item.variant_id: item for item in predictions}
        aligned = [mapping[item.variant_id].fitness_mean for item in candidates if item.variant_id in mapping]
        if not aligned:
            continue
        center = float(np.mean(aligned))
        scale = float(np.std(aligned))
        scale = scale if scale > 1e-8 else 1.0
        for candidate in candidates:
            if candidate.variant_id in mapping:
                accum[candidate.variant_id].append(
                    (float(mapping[candidate.variant_id].fitness_mean) - center) / scale
                )
    return {
        variant_id: float(np.mean(values)) if values else 0.0
        for variant_id, values in accum.items()
    }


class AgentUncertaintySelector:
    """Score Agent proposals without using a fitness predictor by default.

    The GP component is deliberately a coverage model: it uses an RBF kernel over sequence
    Hamming distance and only its posterior variance. It therefore expresses how poorly the
    already observed sequence space covers a proposal; it does not estimate fitness.
    """

    model_version = "kg-llm-agent-uq-v1"

    def __init__(
        self,
        config: GenerationConfig,
        *,
        position_to_index: Mapping[int, int] | None = None,
    ) -> None:
        self.config = config
        self.position_to_index = dict(position_to_index or {})

    def _coverage_uncertainty(
        self,
        observed: Sequence[Variant],
        candidates: Sequence[Variant],
    ) -> dict[str, float]:
        if not observed:
            return {item.variant_id: 1.0 for item in candidates}
        length_scale = self.config.gp_length_scale

        def kernel(left: str, right: str) -> float:
            distance = sum(a != b for a, b in zip(left, right, strict=True))
            return float(np.exp(-(distance**2) / (2.0 * length_scale**2)))

        train = [item.variant for item in observed]
        matrix = np.asarray([[kernel(left, right) for right in train] for left in train])
        matrix += self.config.gp_noise * np.eye(len(train))
        inverse = np.linalg.pinv(matrix, hermitian=True)
        result: dict[str, float] = {}
        for candidate in candidates:
            cross = np.asarray([kernel(candidate.variant, item) for item in train])
            variance = max(0.0, 1.0 - float(cross @ inverse @ cross))
            result[candidate.variant_id] = float(np.sqrt(variance))
        return result

    def score(
        self,
        candidates: Sequence[Variant],
        *,
        observed_variants: Sequence[Variant],
        hypothesis: Hypothesis | None = None,
        hypotheses: Sequence[Hypothesis] = (),
        evidence: Mapping[str, Sequence[Evidence]],
        prior_scores: Mapping[str, float] | None = None,
        predictor_predictions: Sequence[Sequence[Prediction]] = (),
    ) -> list[DesignScore]:
        prior_scores = prior_scores or {}
        active_hypotheses = tuple(hypotheses) or ((hypothesis,) if hypothesis else ())
        uncertainty = self._coverage_uncertainty(observed_variants, candidates)
        if self.config.use_fitness_predictors:
            predictor_scores = _predictor_ensemble_scores(candidates, predictor_predictions)
        else:
            predictor_scores = {item.variant_id: 0.0 for item in candidates}
        output: list[DesignScore] = []
        for variant in candidates:
            hypothesis_values = [
                _hypothesis_score(variant, item, self.position_to_index)
                for item in active_hypotheses
            ]
            if hypothesis_values:
                recency_weights = np.asarray(
                    [
                        self.config.hypothesis_recency_decay
                        ** (len(hypothesis_values) - index - 1)
                        for index in range(len(hypothesis_values))
                    ]
                )
                hypothesis_value = float(
                    np.average(hypothesis_values, weights=recency_weights)
                )
            else:
                hypothesis_value = 0.0
            evidence_value = _evidence_score(evidence.get(variant.variant_id, ()))
            prior_value = float(prior_scores.get(variant.variant_id, 0.0))
            predictor_value = predictor_scores[variant.variant_id]
            utility = (
                self.config.hypothesis_weight * hypothesis_value
                + self.config.evidence_weight * evidence_value
                + self.config.prior_weight * prior_value
                + self.config.uncertainty_beta * uncertainty[variant.variant_id]
                + (
                    self.config.predictor_weight * predictor_value
                    if self.config.use_fitness_predictors
                    else 0.0
                )
            )
            output.append(
                DesignScore(
                    variant_id=variant.variant_id,
                    utility=float(utility),
                    uncertainty=uncertainty[variant.variant_id],
                    hypothesis_score=hypothesis_value,
                    evidence_score=evidence_value,
                    prior_score=prior_value,
                    predictor_score=predictor_value,
                    selection_driver=(
                        "kg_llm_uq_plus_predictor_ensemble"
                        if self.config.use_fitness_predictors
                        else "kg_llm_uq"
                    ),
                    reason=(
                        f"hypothesis={hypothesis_value:.3f}; evidence={evidence_value:.3f}; "
                        f"temporal_prior={prior_value:.3f}; gp_coverage_uncertainty="
                        f"{uncertainty[variant.variant_id]:.3f}; predictor_component="
                        f"{predictor_value:.3f}; fitness_predictors_enabled="
                        f"{self.config.use_fitness_predictors}; hypotheses_considered="
                        f"{len(active_hypotheses)}"
                    ),
                )
            )
        return output

    def as_predictions(self, scores: Sequence[DesignScore]) -> list[Prediction]:
        """Compatibility view for rank/diversity code; these are design utilities, not fitness."""

        return [
            Prediction(
                variant_id=item.variant_id,
                fitness_mean=item.utility,
                fitness_std=item.uncertainty,
                interval_90=(
                    item.utility - 1.645 * item.uncertainty,
                    item.utility + 1.645 * item.uncertainty,
                ),
                ood_score=item.uncertainty,
                component_scores={
                    "hypothesis": item.hypothesis_score,
                    "evidence": item.evidence_score,
                    "temporal_prior": item.prior_score,
                    "predictor_optional": item.predictor_score,
                },
                model_version=self.model_version,
            )
            for item in scores
        ]
