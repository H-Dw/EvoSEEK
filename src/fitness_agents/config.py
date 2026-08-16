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
class GenerationConfig:
    """Mutation-selection controls for Agent modes.

    Fitness predictors remain an explicit optional input, but are disabled by default so the
    knowledge/LLM path can be evaluated independently.
    """

    selection_driver: str = "auto"
    use_fitness_predictors: bool = False
    predictor_models: tuple[ModelConfig, ...] = ()
    hypothesis_weight: float = 1.0
    evidence_weight: float = 0.65
    prior_weight: float = 0.80
    uncertainty_beta: float = 0.75
    predictor_weight: float = 0.0
    gp_length_scale: float = 1.0
    gp_noise: float = 1e-6

    def __post_init__(self) -> None:
        if self.selection_driver not in {"auto", "agent_uq", "predictor", "random"}:
            raise ValueError("generation.selection_driver is invalid")
        if self.gp_length_scale <= 0 or self.gp_noise <= 0:
            raise ValueError("generation GP length scale and noise must be positive")
        if self.predictor_weight < 0:
            raise ValueError("generation.predictor_weight must be non-negative")


@dataclass
class ValidationConfig:
    enabled: bool = True
    predictor_models: tuple[ModelConfig, ...] = ()
    wet_weight: float = 1.0
    dry_weight_cap: float = 0.20
    recency_decay: float = 0.85
    rethink_enabled: bool = True
    dry_reliability_floor: float = 0.05

    def __post_init__(self) -> None:
        if self.wet_weight <= 0:
            raise ValueError("validation.wet_weight must be positive")
        if not 0 <= self.dry_weight_cap < self.wet_weight:
            raise ValueError("validation.dry_weight_cap must be lower than wet_weight")
        if not 0 < self.recency_decay <= 1:
            raise ValueError("validation.recency_decay must be in (0, 1]")
        if not 0 <= self.dry_reliability_floor <= 1:
            raise ValueError("validation.dry_reliability_floor must be in [0, 1]")


@dataclass
class EvaluationConfig:
    metrics: tuple[str, ...] = (
        "spearman",
        "pearson",
        "mse",
        "rmse",
        "ndcg",
        "top_k_hit",
        "top_k_recall",
        "regret_at_k",
        "interval_90_coverage",
        "gaussian_nll",
    )
    top_k: int = 10

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("evaluation.top_k must be positive")
        allowed = {
            "spearman", "pearson", "mse", "rmse", "ndcg", "top_k_hit",
            "top_k_recall", "regret_at_k", "interval_90_coverage", "gaussian_nll",
        }
        unknown = set(self.metrics).difference(allowed)
        if unknown:
            raise ValueError(f"evaluation.metrics contains unsupported values: {sorted(unknown)}")


@dataclass
class OutputConfig:
    artifacts: tuple[str, ...] = (
        "json",
        "csv",
        "markdown",
        "svg",
        "reasoning",
    )
    top_k: int = 10

    def __post_init__(self) -> None:
        allowed = {"json", "csv", "markdown", "svg", "reasoning"}
        unknown = set(self.artifacts).difference(allowed)
        if unknown:
            raise ValueError(f"output.artifacts contains unsupported values: {sorted(unknown)}")
        if self.top_k < 1:
            raise ValueError("output.top_k must be positive")


@dataclass
class KGInteractionRuntimeConfig:
    enabled: bool = True
    enabled_operators: tuple[str, ...] = (
        "hypothesis_context",
        "explain_variant",
        "compare_variants",
    )
    max_tool_calls: int = 3
    max_rows: int = 12
    use_counterevidence: bool = True
    stop_when_sufficient: bool = False

    def __post_init__(self) -> None:
        if self.max_tool_calls < 1 or self.max_rows < 1:
            raise ValueError("kg_interaction limits must be positive")


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
    profile: str = "scientific_v1"
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
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    kg_interaction: KGInteractionRuntimeConfig = field(
        default_factory=KGInteractionRuntimeConfig
    )
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

    def load_model_entries(value: Any) -> tuple[ModelConfig, ...]:
        entries = value or ()
        output: list[ModelConfig] = []
        for entry in entries:
            raw_model = read_yaml(entry, root) if isinstance(entry, (str, Path)) else dict(entry)
            output.append(_dataclass_from_mapping(ModelConfig, raw_model))
        return tuple(output)

    generation_raw = dict(raw.get("generation", {}) or {})
    generation_raw["predictor_models"] = load_model_entries(
        generation_raw.get("predictor_models")
    )
    generation = _dataclass_from_mapping(GenerationConfig, generation_raw)
    validation_raw = dict(raw.get("validation", {}) or {})
    validation_raw["predictor_models"] = load_model_entries(
        validation_raw.get("predictor_models")
    )
    validation = _dataclass_from_mapping(ValidationConfig, validation_raw)
    evaluation_raw = dict(raw.get("evaluation", {}) or {})
    if "metrics" in evaluation_raw:
        evaluation_raw["metrics"] = tuple(str(item) for item in evaluation_raw["metrics"])
    evaluation = _dataclass_from_mapping(EvaluationConfig, evaluation_raw)
    output_raw = dict(raw.get("output", {}) or {})
    if "artifacts" in output_raw:
        output_raw["artifacts"] = tuple(str(item) for item in output_raw["artifacts"])
    output = _dataclass_from_mapping(OutputConfig, output_raw)
    interaction_raw = dict(raw.get("kg_interaction", {}) or {})
    if "enabled_operators" in interaction_raw:
        interaction_raw["enabled_operators"] = tuple(
            str(item) for item in interaction_raw["enabled_operators"]
        )
    kg_interaction = _dataclass_from_mapping(
        KGInteractionRuntimeConfig, interaction_raw
    )
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
        generation=generation,
        validation=validation,
        evaluation=evaluation,
        output=output,
        kg_interaction=kg_interaction,
        output_root=root / raw.get("output_root", "artifacts/runs"),
        llm_provider=llm.provider,
        knowledge_enabled=bool(raw.get("knowledge_enabled", False)),
        score_shuffle=bool(raw.get("score_shuffle", False)),
        evidence_deletion=bool(raw.get("evidence_deletion", False)),
        run_label=str(raw.get("run_label", "")),
        evidence_prefilter_limit=int(raw.get("evidence_prefilter_limit", 5000)),
    )
