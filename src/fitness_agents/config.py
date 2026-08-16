from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CANDIDATE_LIMIT = 64


def _dataclass_from_mapping(cls, raw: dict[str, Any]):
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in allowed})


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
    public_data_path: Path | None = None
    oracle_data_path: Path | None = None
    split_root: Path | None = None
    fold_index: int = 0
    expected_split_strategy: str | None = None
    expected_protocol_version: str | None = None
    expected_manifest_sha256: str | None = None
    expected_assignment_sha256: str | None = None
    sequence_column: str = "variant"
    fitness_column: str = "fitness"
    fitness_scale: str = "raw_assay"
    fitness_transform: str = "identity"

    def __post_init__(self) -> None:
        uses_manifest = self.split_root is not None
        uses_legacy = self.public_data_path is not None or self.oracle_data_path is not None
        if uses_manifest and uses_legacy:
            raise ValueError("Task config cannot mix split_root with legacy public/oracle paths")
        if not uses_manifest and not (
            self.public_data_path is not None and self.oracle_data_path is not None
        ):
            raise ValueError(
                "Task config requires split_root or both public_data_path and oracle_data_path"
            )
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative")


@dataclass
class ModelConfig:
    name: str = "onehot_heterogeneous_ensemble"
    device: str = "cpu"
    allow_device_fallback: bool = False
    batch_size: int = 32
    backend_factory: str | None = None
    checkpoint: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
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
class CriticConfig:
    enabled: bool = True
    mode: str = "rule"
    provider: str = "mock"
    model: str | None = None
    temperature: float = 0.0
    max_revision_attempts: int = 2
    max_model_retries: int = 2
    on_reject: str = "abort_round"
    on_exhausted: str = "abort_round"
    fallback_policy: str = "rule"
    require_counterevidence_search: bool = False
    rationale_visibility: str = "structured_claims_only"
    profile: str = "scientific_v1"
    ood_warning_threshold: float | None = None
    model_disagreement_threshold: float | None = None
    min_batch_distance: int = 1
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    thinking: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"rule", "remote"}:
            raise ValueError("critic.mode must be 'rule' or 'remote'")
        if self.max_revision_attempts not in {0, 1, 2}:
            raise ValueError("critic.max_revision_attempts must be between 0 and 2")
        if self.max_model_retries < 0:
            raise ValueError("critic.max_model_retries must be non-negative")
        if self.on_reject not in {"abort_round", "safe_fallback"}:
            raise ValueError("critic.on_reject must be abort_round or safe_fallback")
        if self.on_exhausted not in {"abort_round", "safe_fallback"}:
            raise ValueError("critic.on_exhausted must be abort_round or safe_fallback")


@dataclass
class LLMConfig:
    provider: str = "mock"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    thinking: str | None = None


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
    critic: CriticConfig = field(default_factory=CriticConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
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
        for key in ("mode", "acquisition", "llm_provider"):
            if key in ablation:
                raw[key] = ablation[key]

    task_values = dict(task_raw)
    for key in ("public_data_path", "oracle_data_path", "split_root"):
        if task_values.get(key):
            path = Path(task_values[key])
            task_values[key] = path if path.is_absolute() else root / path
        else:
            task_values[key] = None
    task = TaskConfig(**task_values)
    profiles = {int(key): value for key, value in knowledge_raw.pop("site_profiles", {}).items()}
    knowledge = KnowledgeConfig(site_profiles=profiles, **knowledge_raw)
    model = _dataclass_from_mapping(ModelConfig, model_raw)
    critic_raw = (
        read_yaml(raw["critic_config"], root)
        if raw.get("critic_config")
        else (raw.get("critic", {}) or {})
    )
    critic = _dataclass_from_mapping(CriticConfig, critic_raw)
    llm_raw: dict[str, Any] = {}
    if raw.get("llm_config"):
        llm_raw.update(read_yaml(raw["llm_config"], root))
    if isinstance(raw.get("llm"), dict):
        llm_raw.update(raw["llm"])
    if raw.get("llm_provider"):
        llm_raw["provider"] = raw["llm_provider"]
    llm = _dataclass_from_mapping(LLMConfig, llm_raw)
    return ExperimentConfig(
        mode=raw["mode"],
        seed=int(raw["seed"]),
        rounds=int(raw["rounds"]),
        budget_per_round=int(raw["budget_per_round"]),
        candidate_limit=int(raw.get("candidate_limit", DEFAULT_CANDIDATE_LIMIT)),
        acquisition=raw["acquisition"],
        ucb_beta=float(raw.get("ucb_beta", 1.5)),
        diversity_lambda=float(raw.get("diversity_lambda", 0.0)),
        task=task,
        model=model,
        knowledge=knowledge,
        critic=critic,
        llm=llm,
        output_root=root / raw.get("output_root", "artifacts/runs"),
        llm_provider=llm.provider,
        knowledge_enabled=bool(raw.get("knowledge_enabled", False)),
        score_shuffle=bool(raw.get("score_shuffle", False)),
        evidence_deletion=bool(raw.get("evidence_deletion", False)),
        run_label=str(raw.get("run_label", "")),
        evidence_prefilter_limit=int(raw.get("evidence_prefilter_limit", 5000)),
    )
