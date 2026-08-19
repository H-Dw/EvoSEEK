from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from fitness_agents.contracts.agent_io import (
    AgentTraceContext,
    RoleActivationState,
    ScientistContextInput,
)
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
        design_space: str = "closed_pool",
        position_policy: str = "configured",
        max_preferred_positions: int = 12,
        allowed_mutation_positions: Sequence[int] | None = None,
    ) -> None:
        self.client = client
        self.task_context = task_context
        self.objective = objective
        self.knowledge_graph = knowledge_graph
        self.design_space = design_space
        self.position_policy = position_policy
        self.max_preferred_positions = max_preferred_positions
        self.allowed_mutation_positions = tuple(
            int(item) for item in (allowed_mutation_positions or task_context.mutable_positions)
        )
        unknown = set(self.allowed_mutation_positions).difference(
            task_context.position_to_sequence_index
        )
        if unknown:
            raise ValueError(
                f"Scientist allowed positions are outside the computation context: {sorted(unknown)}"
            )
        self.last_knowledge_query_id: str | None = None
        self.last_knowledge_query_ids: tuple[str, ...] = ()

    def _default_activation_state(self) -> RoleActivationState:
        open_design = self.design_space == "open_design"
        return RoleActivationState(
            role="scientist",
            design_space="open_design" if open_design else "closed_pool",
            candidate_source=("generated_from_reference" if open_design else "candidate_pool"),
            candidate_pool_consulted=not open_design,
            position_policy=self.position_policy,
            kg_configured=self.knowledge_graph is not None,
            configured_kg_tools=(
                ("hypothesis_context",) if self.knowledge_graph is not None else ()
            ),
        )

    def _activation_state_for_call(
        self,
        activation_state: RoleActivationState | dict[str, Any] | None,
        *,
        evidence: Sequence[Evidence],
        kg_interaction: Any | None,
    ) -> RoleActivationState:
        state = RoleActivationState.model_validate(
            activation_state or self._default_activation_state()
        )
        payload = state.model_dump(mode="python")
        available_channels = {
            item.channel for item in evidence if item.quality_status != "unavailable"
        }
        unavailable_channels = {
            item.channel for item in evidence if item.quality_status == "unavailable"
        }
        executed_tools = tuple(
            dict.fromkeys(pack.operator for pack in getattr(kg_interaction, "packs", ()))
        )
        if not executed_tools and self.knowledge_graph is not None:
            executed_tools = ("hypothesis_context",)
        rag_tools = {"query_local_knowledge", "query_structured_claims"}
        rag_tool_payload_present = any(
            pack.operator in rag_tools
            and bool(getattr(pack, "evidence", ()) or getattr(pack, "facts", ()))
            for pack in getattr(kg_interaction, "packs", ())
        )
        rag_evidence_present = any(item.channel == "local_rag" for item in evidence) or (
            rag_tool_payload_present
        )
        payload.update(
            {
                "role": "scientist",
                "kg_configured": bool(
                    payload["kg_configured"]
                    or kg_interaction is not None
                    or self.knowledge_graph is not None
                ),
                "configured_kg_tools": tuple(
                    dict.fromkeys((*payload["configured_kg_tools"], *executed_tools))
                ),
                "executed_kg_tools": executed_tools,
                "kg_tool_results_present": bool(executed_tools),
                "rag_context_visible": bool(payload["rag_context_visible"] or rag_evidence_present),
                "rag_evidence_present": rag_evidence_present,
                "available_evidence_channels": tuple(sorted(available_channels)),
                "unavailable_evidence_channels": tuple(sorted(unavailable_channels)),
            }
        )
        return RoleActivationState.model_validate(payload)

    def sanitized_context(
        self,
        state: CampaignState,
        observed_variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
    ) -> dict[str, Any]:
        variant_map = {variant.variant_id: variant for variant in observed_variants}
        allowed_text = ",".join(str(item) for item in self.allowed_mutation_positions)
        context = {
            "run_id": state.run_id,
            "mode": state.mode,
            "round_id": state.round_id,
            "expected_hypothesis_id": f"hyp:{state.run_id}:r{state.round_id}",
            "task": (
                f"{self.objective} assay fitness for protein {self.task_context.protein_id}; "
                f"allowed mutation positions are {allowed_text}"
            ),
            "protein_id": self.task_context.protein_id,
            "objective": self.objective,
            "mutable_positions": self.allowed_mutation_positions,
            "allowed_mutation_positions": self.allowed_mutation_positions,
            "sequence_context_scope": (
                "full_reference_sequence"
                if self.design_space == "open_design"
                else "configured_mutable_sites"
            ),
            "computation_position_count": len(self.task_context.mutable_positions),
            "wild_type_sites": "".join(
                self.task_context.full_sequence[
                    self.task_context.position_to_sequence_index[position]
                ]
                for position in self.allowed_mutation_positions
            ),
            "protein_context_id": self.task_context.context_id,
            "design_space": self.design_space,
            "position_policy": self.position_policy,
            "preference_policy": (
                "sparse_subset" if self.design_space == "open_design" else "all_positions"
            ),
            "max_preferred_positions": self.max_preferred_positions,
            "activation_state": self._default_activation_state().model_dump(mode="json"),
            "visible_observations": [
                {
                    "variant_id": observation.variant_id,
                    "variant": variant_map[observation.variant_id].variant,
                    "residues_by_position": {
                        str(position): variant_map[observation.variant_id].sequence[
                            self.task_context.position_to_sequence_index[position]
                        ]
                        for position in self.allowed_mutation_positions
                    },
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
        activation_state: RoleActivationState | dict[str, Any] | None = None,
        approved_subhypotheses: Sequence[dict[str, Any]] = (),
        cross_channel_conflicts: Sequence[dict[str, Any]] = (),
        critic_revision: dict[str, Any] | None = None,
        hypothesis_attempt: int = 0,
    ) -> Hypothesis:
        context = self.sanitized_context(state, observed_variants, observations)
        if hypothesis_attempt > 0:
            context["expected_hypothesis_id"] = (
                f"hyp:{state.run_id}:r{state.round_id}:a{hypothesis_attempt}"
            )
        if approved_subhypotheses:
            approved_payload = list(approved_subhypotheses)
            assert_sanitized(approved_payload, "context.approved_subhypotheses")
            context["approved_subhypotheses"] = approved_payload
        if cross_channel_conflicts:
            conflict_payload = list(cross_channel_conflicts)
            assert_sanitized(conflict_payload, "context.cross_channel_conflicts")
            context["cross_channel_conflicts"] = conflict_payload
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
            self.last_knowledge_query_ids = tuple(pack.query_id for pack in kg_interaction.packs)
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
        context["activation_state"] = self._activation_state_for_call(
            activation_state,
            evidence=evidence,
            kg_interaction=kg_interaction,
        ).model_dump(mode="json")
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
