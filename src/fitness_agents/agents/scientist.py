from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from fitness_agents.contracts.agent_io import AgentTraceContext, ScientistContextInput
from fitness_agents.contracts.interfaces import KnowledgeGraphTool, LLMClient
from fitness_agents.contracts.schemas import (
    CampaignState,
    Evidence,
    FitnessObservation,
    Hypothesis,
    Prediction,
    Variant,
)
from fitness_agents.protein_features import ProteinTaskContext

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

    def __init__(
        self,
        client: LLMClient,
        *,
        task_context: ProteinTaskContext,
        objective: str,
        knowledge_graph: KnowledgeGraphTool | None = None,
    ) -> None:
        self.client = client
        self.task_context = task_context
        self.objective = objective
        self.knowledge_graph = knowledge_graph
        self.last_knowledge_query_id: str | None = None
        self.last_knowledge_query_ids: tuple[str, ...] = ()

    def sanitized_context(
        self,
        state: CampaignState,
        observed_variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
    ) -> dict[str, Any]:
        variant_map = {variant.variant_id: variant for variant in observed_variants}
        context = {
            "run_id": state.run_id,
            "mode": state.mode,
            "round_id": state.round_id,
            "expected_hypothesis_id": f"hyp:{state.run_id}:r{state.round_id}",
            "task": self.task_context.task_description(self.objective),
            "protein_id": self.task_context.protein_id,
            "objective": self.objective,
            "mutable_positions": self.task_context.mutable_positions,
            "wild_type_sites": self.task_context.wild_type_code,
            "protein_context_id": self.task_context.context_id,
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
            "previous_hypothesis_assessment": (
                {
                    "hypothesis_id": state.hypothesis_assessments[-1].hypothesis_id,
                    "status": state.hypothesis_assessments[-1].status.value,
                    "decisive_criterion_ids": list(
                        state.hypothesis_assessments[-1].decisive_criterion_ids
                    ),
                    "unresolved_criterion_ids": list(
                        state.hypothesis_assessments[-1].unresolved_criterion_ids
                    ),
                }
                if state.hypothesis_assessments
                else None
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
        kg_interaction: Any | None = None,
        *,
        critic_revision: dict[str, Any] | None = None,
        hypothesis_attempt: int = 0,
    ) -> Hypothesis:
        context = self.sanitized_context(state, observed_variants, observations)
        if hypothesis_attempt > 0:
            context["expected_hypothesis_id"] = (
                f"hyp:{state.run_id}:r{state.round_id}:a{hypothesis_attempt}"
            )
        if critic_revision is not None:
            assert_sanitized(critic_revision, "context.critic_revision")
            context["critic_revision"] = critic_revision
            rejected_id = critic_revision.get("rejected_hypothesis_id")
            if rejected_id:
                context["previous_hypothesis_id"] = rejected_id
        if kg_interaction is not None:
            interaction_payload = asdict(kg_interaction)
            assert_sanitized(interaction_payload, "context.kg_interaction")
            context["kg_interaction"] = interaction_payload
            self.last_knowledge_query_ids = tuple(
                pack.query_id for pack in kg_interaction.packs
            )
            self.last_knowledge_query_id = (
                self.last_knowledge_query_ids[-1] if self.last_knowledge_query_ids else None
            )
        elif self.knowledge_graph is not None:
            graph_context = self.knowledge_graph.hypothesis_context(round_id=state.round_id)
            assert_sanitized(graph_context, "context.knowledge_graph")
            context["knowledge_graph"] = graph_context
            self.last_knowledge_query_id = str(graph_context["query_id"])
            self.last_knowledge_query_ids = (self.last_knowledge_query_id,)
        else:
            self.last_knowledge_query_id = None
            self.last_knowledge_query_ids = ()
        validated_context = ScientistContextInput.model_validate(context)
        trace_context = AgentTraceContext(
            run_id=state.run_id,
            round_id=state.round_id,
            variant_id=None,
            role="scientist",
            request_id=f"scientist:{state.run_id}:r{state.round_id}:a{hypothesis_attempt}",
            schema_name="HypothesisOutput",
            tool_query_ids=self.last_knowledge_query_ids,
        )
        return self.client.generate_hypothesis(
            sanitized_context=validated_context,
            evidence=evidence,
            output_schema=HYPOTHESIS_SCHEMA,
            trace_context=trace_context.model_dump(mode="json"),
        )

    def inspect_variant(self, variant_id: str, *, round_id: int) -> dict[str, Any]:
        """Expose the same safe KG interface to future critic/designer agent steps."""
        if self.knowledge_graph is None:
            raise RuntimeError("Knowledge graph tool is not configured for this agent")
        context = self.knowledge_graph.explain_variant(variant_id, round_id=round_id)
        assert_sanitized(context, "context.knowledge_graph_variant")
        self.last_knowledge_query_id = str(context["query_id"])
        return context

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
