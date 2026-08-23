from __future__ import annotations

import pytest

from fitness_agents.agents.client_registry import ClientRegistry
from fitness_agents.agents.output_contracts import validate_hypothesis_rethink_payload
from fitness_agents.agents.profile_loader import load_role_profile
from fitness_agents.agents.rethink import RETHINK_DIMENSION_GROUPS
from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.contracts.agent_io import RoleActivationState


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


def test_rethink_output_requires_all_four_groups_and_eight_dimensions() -> None:
    payload = {
        "hypothesis_id": "H01",
        "assessment_id": "HA01",
        "assessment_status": "SUPPORTED",
        "dimension_groups": [
            {
                "hypothesis_id": "H01",
                "assessment_id": "HA01",
                "group_name": group_name,
                "dimension_assessments": [
                    {
                        "dimension": dimension,
                        "evidence_status": "measured",
                        "relation_to_hypothesis": "positive",
                        "finding_code": "supported",
                        "finding": "The criterion receipt supports this bounded dimension.",
                        "implication": "Retain the claim only within the tested scope.",
                    }
                    for dimension in dimensions
                ],
                "group_advice": "Keep the conclusion bounded to the assessed hypothesis.",
            }
            for group_name, dimensions in RETHINK_DIMENSION_GROUPS.items()
        ],
    }
    payload["dimension_groups"].pop()

    with pytest.raises(ValueError, match="at least 4 items"):
        validate_hypothesis_rethink_payload(
            payload,
            expected_hypothesis_id="H01",
            expected_assessment_id="HA01",
            expected_assessment_status="SUPPORTED",
        )


def test_client_registry_is_allowlisted() -> None:
    registry = ClientRegistry()
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        registry.create("unregistered")
