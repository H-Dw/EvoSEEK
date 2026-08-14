from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing pyproject.toml")


def read_yaml(path: str | Path, root: Path | None = None) -> dict[str, Any]:
    base = root or project_root()
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = base / file_path
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class TaskConfig:
    task_id: str
    protein_id: str
    assay_id: str
    wild_type_sites: str
    mutable_positions: list[int]
    objective: str
    public_data_path: Path
    oracle_data_path: Path
    sequence_column: str = "variant"
    fitness_column: str = "fitness"


@dataclass
class ModelConfig:
    name: str = "onehot_heterogeneous_ensemble"
    feature_provider: str = "gb1_onehot_pairwise"
    ridge_members: int = 5
    extra_trees_estimators: int = 160
    ridge_alpha: float = 10.0
    bootstrap_fraction: float = 0.85
    conformal_alpha: float = 0.10
    include_gaussian_process: bool = False


@dataclass
class KnowledgeConfig:
    physchem: bool = True
    conservation: bool = True
    structure: bool = True
    kg: bool = True
    soft_weight: float = 0.20
    site_profiles: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    mode: str
    seed: int
    rounds: int
    budget_per_round: int
    candidate_limit: int
    acquisition: str
    ucb_beta: float
    diversity_lambda: float
    task: TaskConfig
    model: ModelConfig
    knowledge: KnowledgeConfig
    output_root: Path
    llm_provider: str = "mock"
    knowledge_enabled: bool = False
    score_shuffle: bool = False
    evidence_deletion: bool = False
    run_label: str = ""
    evidence_prefilter_limit: int = 5000


def load_experiment_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
    ablation_path: str | Path | None = None,
) -> ExperimentConfig:
    root = project_root()
    raw = read_yaml(path, root)
    if overrides:
        raw.update(overrides)
    task_raw = read_yaml(raw["task_config"], root)
    model_raw = read_yaml(raw["model_config"], root)
    knowledge_raw = read_yaml(raw["knowledge_config"], root)
    if ablation_path:
        ablation = read_yaml(ablation_path, root)
        for key in ("physchem", "conservation", "structure", "kg"):
            if key in ablation:
                knowledge_raw[key] = bool(ablation[key])
        for key in ("mode", "acquisition"):
            if key in ablation:
                raw[key] = ablation[key]

    task = TaskConfig(
        **{
            **task_raw,
            "public_data_path": root / task_raw["public_data_path"],
            "oracle_data_path": root / task_raw["oracle_data_path"],
        }
    )
    profiles = {int(key): value for key, value in knowledge_raw.pop("site_profiles", {}).items()}
    knowledge = KnowledgeConfig(site_profiles=profiles, **knowledge_raw)
    model = ModelConfig(**model_raw)
    return ExperimentConfig(
        mode=raw["mode"],
        seed=int(raw["seed"]),
        rounds=int(raw["rounds"]),
        budget_per_round=int(raw["budget_per_round"]),
        candidate_limit=int(raw.get("candidate_limit", 0)),
        acquisition=raw["acquisition"],
        ucb_beta=float(raw.get("ucb_beta", 1.5)),
        diversity_lambda=float(raw.get("diversity_lambda", 0.0)),
        task=task,
        model=model,
        knowledge=knowledge,
        output_root=root / raw.get("output_root", "artifacts/runs"),
        llm_provider=raw.get("llm_provider", "mock"),
        knowledge_enabled=bool(raw.get("knowledge_enabled", False)),
        score_shuffle=bool(raw.get("score_shuffle", False)),
        evidence_deletion=bool(raw.get("evidence_deletion", False)),
        run_label=str(raw.get("run_label", "")),
        evidence_prefilter_limit=int(raw.get("evidence_prefilter_limit", 5000)),
    )
