from __future__ import annotations

import pytest

from fitness_agents.agents.client_registry import ClientRegistry
from fitness_agents.agents.output_contracts import validate_rethink_payload
from fitness_agents.agents.profile_loader import load_role_profile
from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.contracts.agent_io import ReThinkContextInput


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
    assert scientist.sha256 != rethink.sha256


def test_rethink_context_and_output_require_exact_candidate_coverage() -> None:
    context = ReThinkContextInput.model_validate(
        {"run_id": "run", "round_id": 1, "visible_baseline": 0.0,
         "candidates": [{"variant_id": "v1"}, {"variant_id": "v2"}]}
    )
    payload = {
        "reflections": [{
            "variant_id": "v1", "verdict": "support", "summary": "supported",
            "positive_findings": [], "negative_findings": [],
            "revised_reason": "bounded reason", "next_round_advice": "test again",
        }]
    }
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_rethink_payload(payload, expected_variant_ids=context.expected_variant_ids)


def test_client_registry_is_allowlisted() -> None:
    registry = ClientRegistry()
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        registry.create("unregistered")
