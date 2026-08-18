from dataclasses import replace

import pytest

from fitness_agents.config import TaskConfig, load_experiment_config


def test_standard_al96_config_uses_manifest_fold_source():
    config = load_experiment_config("configs/experiments/knowledge_agent_al96.yaml")
    assert config.task.split_root is not None
    assert config.task.public_data_path is None
    assert config.task.oracle_data_path is None
    assert config.task.fold_index == 0
    assert config.task.expected_split_strategy == "al96_closed_loop"
    assert config.condition == "knowledge_agent"
    assert config.knowledge.local_knowledge.enabled is False
    assert config.knowledge.local_knowledge.allow_remote_context is False


def test_al96_rag_config_enables_remote_local_knowledge_without_changing_mode():
    no_rag = load_experiment_config("configs/experiments/knowledge_agent_al96.yaml")
    rag = load_experiment_config("configs/experiments/knowledge_agent_al96_rag.yaml")
    assert rag.mode == no_rag.mode == "knowledge_agent"
    assert rag.condition == "knowledge_agent_rag"
    assert rag.knowledge.local_knowledge.enabled is True
    assert rag.knowledge.local_knowledge.allow_remote_context is True
    assert rag.kg_interaction.max_tool_calls >= no_rag.kg_interaction.max_tool_calls
    assert rag.task.split_root == no_rag.task.split_root
    assert rag.generation.selection_driver == no_rag.generation.selection_driver


def test_agent_experiments_wire_deepseek_and_kermut_without_literal_secrets():
    knowledge = load_experiment_config("configs/experiments/knowledge_agent.yaml")
    llm_agent = load_experiment_config("configs/experiments/llm_agent.yaml")
    random_run = load_experiment_config("configs/experiments/random.yaml")
    fitness = load_experiment_config("configs/experiments/fitness_direct.yaml")

    for config in (knowledge, llm_agent):
        assert config.llm.provider == "deepseek"
        assert config.llm.model == "deepseek-v4-flash"
        assert config.llm.api_key == "env:DEEPSEEK_API_KEY"
        assert config.llm.base_url == "https://api.deepseek.com"
        assert config.critic.mode == "remote"
        assert config.critic.api_key == "env:DEEPSEEK_API_KEY"
        assert config.model.name == "kermut"
        assert config.model.options["conditional_probs_path"] == (
            "models/kermut/gb1_sites_conditional_probs.npy"
        )
        assert config.model.checkpoint == (
            "~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt"
        )

    assert random_run.llm.provider == "mock"
    assert fitness.llm.provider == "mock"
    assert random_run.model.name == "kermut"
    assert fitness.model.name == "kermut"
    assert random_run.critic.mode == "rule"


def test_no_llm_ablation_forces_mock_provider():
    config = load_experiment_config(
        "configs/experiments/knowledge_agent.yaml",
        ablation_path="configs/ablation/no_llm.yaml",
    )
    assert config.mode == "fitness_direct"
    assert config.llm.provider == "mock"


def test_task_config_rejects_ambiguous_manifest_and_legacy_sources(experiment_config):
    with pytest.raises(ValueError, match="cannot mix"):
        replace(experiment_config.task, split_root=experiment_config.output_root)


def test_task_config_requires_one_complete_source_mode():
    with pytest.raises(ValueError, match="requires split_root"):
        TaskConfig(
            task_id="invalid",
            protein_id="GB1",
            assay_id="invalid",
            wild_type_sites="VDGV",
            mutable_positions=[39, 40, 41, 54],
            objective="maximize",
        )
