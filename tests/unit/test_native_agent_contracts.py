from __future__ import annotations

import pytest

from fitness_agents.agents.client_registry import ClientRegistry
from fitness_agents.agents.output_contracts import validate_rethink_payload
from fitness_agents.agents.profile_loader import load_role_profile
from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.contracts.agent_io import ReThinkContextInput, RoleActivationState


def test_removed_agents_runtime_fails_fast() -> None:
    with pytest.raises(ValueError, match="Removed Agents SDK runtime"):
        load_experiment_config(
            project_root() / "configs/experiments/knowledge_agent.yaml",
            overrides={"llm": {"runtime": "agents_sdk"}},
        )


def test_role_profiles_are_versioned_and_authority_free() -> None:
    scientist = load_role_profile("scientist", "scientific_v1")
    rethink = load_role_profile("rethink", "scientific_v1")
    assert scientist.metadata["state_authority"] == "none"
    assert rethink.metadata["tool_policy"] == "none"
    assert scientist.name == rethink.name == "scientific_v1"
    assert scientist.role != rethink.role
    assert scientist.metadata["version"] != rethink.metadata["version"]
    assert "Activation-state routing" in scientist.instructions
    assert "Activation-state routing" in rethink.instructions
    for instructions in (scientist.instructions, rethink.instructions):
        for forbidden_prior in ("AAIndex", "Neff", "SASA", "salt-bridge", "hydropathy"):
            assert forbidden_prior not in instructions


def test_role_activation_state_distinguishes_configured_and_executed_routes() -> None:
    state = RoleActivationState.model_validate(
        {
            "role": "scientist",
            "design_space": "open_design",
            "candidate_source": "generated_from_reference",
            "candidate_pool_consulted": False,
            "position_policy": "all",
            "selection_driver": "active_learning",
            "active_learning_enabled": True,
            "fitness_predictors_used_for_generation": True,
            "rag_configured": True,
            "rag_context_visible": True,
            "rag_retrieval_performed": True,
            "rag_evidence_present": False,
            "kg_configured": True,
            "kg_interaction_enabled": True,
            "configured_kg_tools": ["hypothesis_context", "query_local_knowledge"],
            "executed_kg_tools": ["hypothesis_context"],
            "kg_tool_results_present": True,
            "available_evidence_channels": ["kg"],
            "unavailable_evidence_channels": ["structure"],
        }
    )

    assert state.design_space == "open_design"
    assert state.candidate_pool_consulted is False
    assert state.configured_kg_tools != state.executed_kg_tools


def test_rethink_context_and_output_require_exact_candidate_coverage() -> None:
    context = ReThinkContextInput.model_validate(
        {
            "run_id": "run",
            "round_id": 1,
            "visible_baseline": 0.0,
            "baseline_receipt": {
                "value": 0.0,
                "statistic": "pre_round_visible_median",
                "source": "revealed_observations_before_current_round",
            },
            "measurement_contract": {
                "assay_id": "assay:test",
                "fitness_scale": "raw_assay",
                "optimization_direction": "higher_is_better",
            },
            "final_critic_decision": {
                "decision_id": "D01-00",
                "verdict": "APPROVE",
                "summary": "approved",
                "cited_evidence_ids": [],
            },
            "candidates": [
                {
                    "variant_id": item,
                    "mutation_notation": "V39A",
                    "agent_reason": "bounded",
                    "evidence_ids": [],
                    "wet_value": 0.1,
                    "dry_validations": [],
                    "intent_arm": "coverage_exploration",
                    "allow_hypothesis_mismatch": False,
                    "falsification_role": "not_in_primary_criterion",
                }
                for item in ("v1", "v2")
            ],
        }
    )
    payload = {
        "reflections": [
            {
                "variant_id": "v1",
                "candidate_relation": "support",
                "summary": "supported",
                "positive_findings": [],
                "negative_findings": [],
                "revised_reason": "bounded reason",
                "next_round_advice": "test again",
                "next_round_action": "test_matched_alternative",
            }
        ],
        "batch_assessment": {
            "assessment_id": None,
            "status": "NOT_APPLICABLE",
            "commentary": "No batch assessment applies.",
            "next_round_advice": "Keep candidate relations bounded.",
        },
    }
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_rethink_payload(payload, expected_variant_ids=context.expected_variant_ids)


def test_client_registry_is_allowlisted() -> None:
    registry = ClientRegistry()
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        registry.create("unregistered")
