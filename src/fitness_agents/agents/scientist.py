from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fitness_agents.contracts.interfaces import LLMClient
from fitness_agents.contracts.schemas import (
    CampaignState,
    Evidence,
    FitnessObservation,
    Hypothesis,
    Prediction,
    Variant,
)

from .llm import HYPOTHESIS_SCHEMA

FORBIDDEN_CONTEXT_KEYS = {
    "raw_fitness",
    "normalized_fitness",
    "oracle_path",
    "oracle_data_path",
    "final_test",
    "final_test_ids",
}


def assert_sanitized(value: Any, path: str = "context") -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_CONTEXT_KEYS.intersection(value)
        if forbidden:
            raise ValueError(f"Forbidden hidden-label keys at {path}: {sorted(forbidden)}")
        for key, item in value.items():
            assert_sanitized(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_sanitized(item, f"{path}[{index}]")


class ScientistAgent:
    """Hypothesis/critic layer that never receives hidden oracle labels."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @staticmethod
    def sanitized_context(
        state: CampaignState,
        observed_variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
    ) -> dict[str, Any]:
        variant_map = {variant.variant_id: variant for variant in observed_variants}
        context = {
            "run_id": state.run_id,
            "mode": state.mode,
            "round_id": state.round_id,
            "task": "maximize GB1 IgG-binding fitness over sites 39,40,41,54",
            "visible_observations": [
                {
                    "variant_id": observation.variant_id,
                    "variant": variant_map[observation.variant_id].variant,
                    "mutation_notation": variant_map[observation.variant_id].mutation_notation,
                    "measured_fitness": observation.fitness,
                    "round_revealed": observation.round_revealed,
                }
                for observation in observations
            ],
            "previous_hypothesis_id": (
                state.hypotheses[-1].hypothesis_id if state.hypotheses else None
            ),
        }
        assert_sanitized(context)
        return context

    def propose_hypothesis(
        self,
        state: CampaignState,
        observed_variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
        evidence: Sequence[Evidence],
    ) -> Hypothesis:
        context = self.sanitized_context(state, observed_variants, observations)
        return self.client.generate_hypothesis(
            sanitized_context=context,
            evidence=evidence,
            output_schema=HYPOTHESIS_SCHEMA,
        )

    @staticmethod
    def critique(
        variant: Variant,
        prediction: Prediction,
        evidence: Sequence[Evidence],
        hypothesis: Hypothesis | None,
        intervention_tags: Sequence[str],
    ) -> str:
        evidence_channels = sorted({entry.channel for entry in evidence})
        hypothesis_text = hypothesis.hypothesis_id if hypothesis else "none"
        intervention_text = ",".join(intervention_tags) if intervention_tags else "none"
        return (
            f"selected under hypothesis={hypothesis_text}; predicted mean={prediction.fitness_mean:.4f}, "
            f"epistemic/calibrated std={prediction.fitness_std:.4f}, OOD={prediction.ood_score:.3f}; "
            f"evidence_channels={evidence_channels}; interventions={intervention_text}. "
            "Prediction is not a measurement and the hypothesis is tested only after oracle reveal."
        )

