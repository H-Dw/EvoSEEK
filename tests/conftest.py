from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from fitness_agents.config import ExperimentConfig, KnowledgeConfig, ModelConfig, TaskConfig
from fitness_agents.data.gb1 import canonical_mutation_notation, variant_id


def _synthetic_fitness(code: str) -> float:
    preferred = ("F", "W", "A", "L")
    additive = sum(0.35 for actual, target in zip(code, preferred, strict=True) if actual == target)
    wt_bonus = sum(0.06 for actual, wt in zip(code, "VDGV", strict=True) if actual == wt)
    epistasis = 0.8 if code[0] == "F" and code[1] == "W" else 0.0
    return 0.1 + additive + wt_bonus + epistasis


@pytest.fixture
def synthetic_benchmark(tmp_path: Path) -> dict[str, Path]:
    alphabet = "VDGVAWFL"
    codes = []
    for values in product(alphabet, repeat=4):
        code = "".join(values)
        if code not in codes:
            codes.append(code)
        if len(codes) == 144:
            break
    rows = []
    roles = ["initial_observed"] * 24 + ["validation"] * 16 + ["oracle_pool"] * 88 + ["final_test"] * 16
    for code, role in zip(codes, roles, strict=True):
        rows.append(
            {
                "variant_id": variant_id(code),
                "variant": code,
                "sequence": code,
                "mutation_notation": canonical_mutation_notation(code),
                "mutation_count": sum(a != b for a, b in zip(code, "VDGV", strict=True)),
                "split_role": role,
                "fitness": _synthetic_fitness(code),
            }
        )
    # Sentinel hidden label used by leakage tests.
    rows[-1]["fitness"] = 98765.4321
    frame = pd.DataFrame(rows)
    public = tmp_path / "public.csv"
    oracle = tmp_path / "oracle.csv"
    frame.drop(columns="fitness").to_csv(public, index=False)
    frame[["variant_id", "fitness", "split_role"]].to_csv(oracle, index=False)
    return {"public": public, "oracle": oracle, "root": tmp_path}


@pytest.fixture
def experiment_config(synthetic_benchmark: dict[str, Path]) -> ExperimentConfig:
    task = TaskConfig(
        task_id="synthetic_gb1",
        protein_id="GB1",
        assay_id="synthetic",
        wild_type_sites="VDGV",
        mutable_positions=[39, 40, 41, 54],
        objective="maximize",
        public_data_path=synthetic_benchmark["public"],
        oracle_data_path=synthetic_benchmark["oracle"],
    )
    model = ModelConfig(
        ridge_members=3,
        extra_trees_estimators=20,
        bootstrap_fraction=0.8,
    )
    knowledge = KnowledgeConfig(
        site_profiles={
            39: {"wild_type": "V", "tolerated": ["V", "F"], "structure_risk": 0.2},
            40: {"wild_type": "D", "tolerated": ["D", "W"], "structure_risk": 0.3},
            41: {"wild_type": "G", "tolerated": ["G", "A"], "structure_risk": 0.7},
            54: {"wild_type": "V", "tolerated": ["V", "L"], "structure_risk": 0.3},
        }
    )
    return ExperimentConfig(
        mode="knowledge_agent",
        seed=7,
        rounds=2,
        budget_per_round=4,
        candidate_limit=40,
        acquisition="greedy",
        ucb_beta=1.5,
        diversity_lambda=0.1,
        task=task,
        model=model,
        knowledge=knowledge,
        output_root=synthetic_benchmark["root"] / "runs",
        llm_provider="mock",
        knowledge_enabled=True,
    )


@pytest.fixture
def config_factory(experiment_config: ExperimentConfig):
    def factory(**changes):
        return replace(experiment_config, **changes)

    return factory

