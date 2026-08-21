from __future__ import annotations

import pytest

from fitness_agents.config import (
    GenerationConfig,
    PriorScheduleConfig,
    load_experiment_config,
)


def test_prior_schedule_defaults_to_upfront():
    schedule = PriorScheduleConfig()
    assert schedule.mode == "upfront"
    assert schedule.keep_wild_type is True


def test_prior_schedule_rejects_unknown_mode():
    with pytest.raises(ValueError, match="prior_schedule.mode"):
        PriorScheduleConfig(mode="delayed")


def test_mutation_order_schedule_defaults_to_unrestricted():
    assert GenerationConfig().mutation_order_schedule == {}


def test_mutation_order_schedule_normalizes_keys_and_orders():
    config = GenerationConfig(mutation_order_schedule={"1": [3, 1, 1], 2: (2,)})
    assert config.mutation_order_schedule == {1: (1, 3), 2: (2,)}


@pytest.mark.parametrize(
    "schedule",
    [{0: [1]}, {1: [0]}, {1: []}, {-2: [1]}],
)
def test_mutation_order_schedule_rejects_invalid_entries(schedule):
    with pytest.raises(ValueError, match="mutation_order_schedule"):
        GenerationConfig(mutation_order_schedule=schedule)


def test_default_experiment_config_keeps_upfront_prior_and_no_order_schedule():
    config = load_experiment_config("configs/experiments/hierarchical_scientist.deepseek.yaml")
    assert config.prior_schedule.mode == "upfront"
    assert config.prior_schedule.keep_wild_type is True
    assert config.generation.mutation_order_schedule == {}


def test_coldstart_experiment_config_parses_prior_and_order_schedule():
    config = load_experiment_config(
        "configs/experiments/hierarchical_scientist.coldstart.deepseek.yaml"
    )
    assert config.prior_schedule.mode == "cold_start"
    assert config.prior_schedule.keep_wild_type is True
    assert config.generation.mutation_order_schedule == {1: (1,)}
    assert config.task.expected_protocol_version == "GB1-AL96-5CV-v2"
