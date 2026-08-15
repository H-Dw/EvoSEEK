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
