import pytest

from fitness_agents.agents.critic import load_critic_profile
from fitness_agents.agents.profile_loader import load_role_profile
from fitness_agents.config import (
    HierarchicalHypothesisConfig,
    LLMConfig,
    ValidationConfig,
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
    assert config.llm.model == "deepseek-v4-pro"
    assert config.llm.max_transport_retries == 2
    assert config.llm.max_truncation_retries == 1
    assert config.llm.max_schema_retries == 2
    assert config.llm.max_semantic_retries == 1
    assert config.critic.max_model_retries == 2
    assert config.critic.max_tokens == 32768
    assert config.critic.max_truncation_retries == 1
    assert config.critic.max_schema_retries == 2
    assert config.llm.max_input_chars == 160000
    assert config.llm.max_tokens == 32768
    assert config.llm.rethink_max_tokens == 49152
    assert config.llm.rethink_render_max_tokens == 32768
    assert config.llm.rethink_reasoning_effort is None
    assert config.llm.rethink_thinking == "disabled"
    assert config.llm.rethink_reasoning_batch_size == 1
    assert config.llm.rethink_max_parallel_batches == 8
    assert config.llm.rethink_max_calls_per_round == 256
    assert config.llm.rethink_call_reserve == 96
    assert config.llm.rethink_dimension_parallel is True
    assert config.llm.rethink_parallel_dimension_groups is True
    assert config.scientist_prompt_evidence_limit == 32
    assert config.hierarchical_hypothesis.main_max_input_chars == 160000
    assert config.hierarchical_hypothesis.child_max_input_chars == 120000
    assert config.hierarchical_hypothesis.critic_max_input_chars == 120000
    assert config.hierarchical_hypothesis.subcritic_mode == "remote"
    assert config.hierarchical_hypothesis.child_max_tokens == 32768
    assert config.hierarchical_hypothesis.child_critic_max_tokens == 32768
    assert config.hierarchical_hypothesis.main_critic_max_tokens == 32768
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
    for role, profile in (
        ("scientist", "scientific_v1"),
        ("scientist", "synthesis_v1"),
        ("critic", "hypothesis_v1"),
        ("rethink", "scientific_v1"),
        ("rethink", "sample_v1"),
        ("subscientist", "physchem_v1"),
        ("subcritic", "physchem_v1"),
    ):
        assert "Shared ID Integrity Protocol" in load_role_profile(
            role, profile
        ).instructions
    assert "Shared ID Integrity Protocol" in load_critic_profile("scientific_v1")


def test_formal_retry_caps_reject_unbounded_configuration() -> None:
    with pytest.raises(ValueError, match="between 0 and 2"):
        LLMConfig(max_transport_retries=3)
    with pytest.raises(ValueError, match="between 0 and 2"):
        HierarchicalHypothesisConfig(max_child_revision_attempts=3)
    with pytest.raises(ValueError, match="input budgets"):
        HierarchicalHypothesisConfig(child_max_input_chars=4000)
    with pytest.raises(ValueError, match="max_tokens"):
        LLMConfig(max_tokens=131073)
    with pytest.raises(ValueError, match="max_parallel_batches"):
        LLMConfig(rethink_max_parallel_batches=17)
    with pytest.raises(ValueError, match="rethink_max_tokens"):
        LLMConfig(rethink_max_tokens=131073)
    with pytest.raises(ValueError, match="child_sample_batch_size"):
        HierarchicalHypothesisConfig(child_sample_batch_size=9)


def test_hypothesis_rethink_is_an_explicit_opt_in() -> None:
    assert ValidationConfig().rethink_mode == "sample"
    assert ValidationConfig(rethink_mode="hypothesis").rethink_mode == "hypothesis"
    with pytest.raises(ValueError, match="rethink_mode"):
        ValidationConfig(rethink_mode="batch")

    legacy = load_experiment_config("configs/experiments/knowledge_agent.yaml")
    grouped = load_experiment_config(
        "configs/experiments/gb1_rethink_hypothesis_smoke.yaml"
    )
    assert legacy.validation.rethink_mode == "sample"
    assert grouped.validation.rethink_mode == "hypothesis"
