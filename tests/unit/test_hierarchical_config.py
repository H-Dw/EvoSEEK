import pytest

from fitness_agents.agents.profile_loader import load_role_profile
from fitness_agents.config import (
    HierarchicalHypothesisConfig,
    LLMConfig,
    load_experiment_config,
)


def test_formal_feature_route_enables_hierarchy_and_bounded_retries() -> None:
    config = load_experiment_config(
        "configs/experiments/knowledge_agent_features.deepseek.example.yaml"
    )
    assert config.hierarchical_hypothesis.enabled is True
    assert config.hierarchical_hypothesis.max_parallel_branches == 3
    assert config.hierarchical_hypothesis.max_child_revision_attempts == 1
    assert config.hierarchical_hypothesis.max_main_revision_attempts == 1
    assert config.llm.model == "deepseek-v4-flash"
    assert config.llm.max_transport_retries == 2
    assert config.llm.max_output_retries == 1
    assert config.critic.max_model_retries == 2
    assert config.critic.max_output_retries == 1
    assert config.llm.max_input_chars == 80000
    assert config.llm.allow_unknown_evidence_stripping is False
    smoke = load_experiment_config(
        "configs/experiments/knowledge_agent_features.example.yaml"
    )
    assert smoke.llm.provider == "mock"
    assert smoke.hierarchical_hypothesis.enabled is True


def test_all_hierarchical_role_profiles_are_versioned_and_loadable() -> None:
    for channel in ("physchem", "conservation", "structure"):
        scientist = load_role_profile("subscientist", f"{channel}_v1")
        critic = load_role_profile("subcritic", f"{channel}_v1")
        assert channel in scientist.instructions.lower()
        assert channel in critic.instructions.lower()
        assert scientist.sha256 and critic.sha256
    assert load_role_profile("scientist", "synthesis_v1").sha256
    assert load_role_profile("critic", "hypothesis_v1").sha256


def test_formal_retry_caps_reject_unbounded_configuration() -> None:
    with pytest.raises(ValueError, match="between 0 and 2"):
        LLMConfig(max_transport_retries=3)
    with pytest.raises(ValueError, match="between 0 and 2"):
        HierarchicalHypothesisConfig(max_child_revision_attempts=3)
