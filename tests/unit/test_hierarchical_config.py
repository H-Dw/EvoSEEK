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
    assert config.hierarchical_hypothesis.child_sample_batch_size == 8
    assert config.hierarchical_hypothesis.child_max_parallel_batches == 2
    assert config.hierarchical_hypothesis.max_child_revision_attempts == 1
    assert config.hierarchical_hypothesis.max_main_revision_attempts == 2
    assert config.llm.model == "deepseek-v4-flash"
    assert config.llm.max_transport_retries == 2
    assert config.llm.max_truncation_retries == 1
    assert config.llm.max_schema_retries == 2
    assert config.llm.max_semantic_retries == 1
    assert config.critic.max_model_retries == 2
    assert config.critic.max_tokens == 20000
    assert config.critic.max_truncation_retries == 1
    assert config.critic.max_schema_retries == 2
    assert config.llm.max_input_chars == 160000
    assert config.llm.max_tokens == 20000
    assert config.llm.rethink_max_tokens == 20000
    assert config.llm.rethink_reasoning_batch_size == 1
    assert config.llm.rethink_max_parallel_batches == 8
    assert config.llm.rethink_max_calls_per_round == 160
    assert config.llm.rethink_call_reserve == 80
    assert config.llm.rethink_dimension_parallel is True
    assert config.scientist_prompt_evidence_limit == 32
    assert config.hierarchical_hypothesis.main_max_input_chars == 160000
    assert config.hierarchical_hypothesis.child_max_input_chars == 120000
    assert config.hierarchical_hypothesis.critic_max_input_chars == 120000
    assert config.hierarchical_hypothesis.subcritic_mode == "remote"
    assert config.hierarchical_hypothesis.child_max_tokens == 20000
    assert config.hierarchical_hypothesis.child_critic_max_tokens == 20000
    assert config.hierarchical_hypothesis.main_critic_max_tokens == 20000
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
        assert "sha256" not in scientist.instructions.casefold()
        assert "sha256" not in critic.instructions.casefold()
    assert "sha256" not in load_role_profile(
        "scientist", "synthesis_v1"
    ).instructions.casefold()
    assert "sha256" not in load_role_profile(
        "critic", "hypothesis_v1"
    ).instructions.casefold()


def test_formal_retry_caps_reject_unbounded_configuration() -> None:
    with pytest.raises(ValueError, match="between 0 and 2"):
        LLMConfig(max_transport_retries=3)
    with pytest.raises(ValueError, match="between 0 and 2"):
        HierarchicalHypothesisConfig(max_child_revision_attempts=3)
    with pytest.raises(ValueError, match="input budgets"):
        HierarchicalHypothesisConfig(child_max_input_chars=4000)
    with pytest.raises(ValueError, match="max_tokens"):
        LLMConfig(max_tokens=20001)
    with pytest.raises(ValueError, match="reasoning_batch_size"):
        LLMConfig(rethink_reasoning_batch_size=9)
    with pytest.raises(ValueError, match="rethink_max_tokens"):
        LLMConfig(rethink_max_tokens=20001)
    with pytest.raises(ValueError, match="child_sample_batch_size"):
        HierarchicalHypothesisConfig(child_sample_batch_size=9)
