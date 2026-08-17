from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CANDIDATE_LIMIT = 64
REMOVED_SDK_LLM_KEYS = frozenset(
    {"agents_sdk", "sdk_tracing_enabled", "sdk_max_turns", "sdk_model_retries"}
)


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
    reference_sequence: str | None = None
    reference_sequence_path: Path | None = None
    sequence_position_offset: int = 1
    numbering_scheme: str = "task"
    assay_conditions: dict[str, Any] = field(default_factory=dict)
    structure_resources: tuple[dict[str, Any], ...] = ()
    protein_name: str | None = None
    protein_aliases: tuple[str, ...] = ()
    protein_accessions: tuple[str, ...] = ()

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
        if self.reference_sequence and self.reference_sequence_path:
            raise ValueError(
                "Task config cannot set both reference_sequence and reference_sequence_path"
            )
        if self.sequence_position_offset < 0:
            raise ValueError("sequence_position_offset must be non-negative")
        self.protein_aliases = tuple(str(item) for item in self.protein_aliases)
        self.protein_accessions = tuple(str(item) for item in self.protein_accessions)


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


@dataclass(frozen=True)
class LearnableParameterSpec:
    value: float
    category: str = "scientific"
    status: str = "expert_prior"
    learnable: bool = False
    bounds: tuple[float, float] | None = None
    transform: str = "identity"
    update_policy: str = "never"
    min_evidence: int = 0
    source: str = "unspecified"
    version: str = "v1"

    def __post_init__(self) -> None:
        if self.category not in {"scientific", "policy", "operational"}:
            raise ValueError(f"Unknown parameter category: {self.category}")
        if self.transform not in {"identity", "log", "logit"}:
            raise ValueError(f"Unknown parameter transform: {self.transform}")
        if self.min_evidence < 0:
            raise ValueError("min_evidence must be non-negative")
        if self.bounds is not None:
            lower, upper = self.bounds
            if lower >= upper or not lower <= self.value <= upper:
                raise ValueError("Parameter value must lie inside increasing bounds")


@dataclass(frozen=True)
class KnowledgeProviderConfig:
    kind: str
    enabled: bool = True
    resource_path: Path | None = None
    contributes_to_selection: bool = False
    calibration: str = "none"
    minimum_calibration_samples: int = 8
    missing_policy: str = "unavailable"
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.calibration not in {"none", "visible_linear"}:
            raise ValueError(f"Unsupported evidence calibration: {self.calibration}")
        if self.minimum_calibration_samples < 2:
            raise ValueError("minimum_calibration_samples must be at least 2")
        if self.missing_policy not in {"unavailable", "fail"}:
            raise ValueError("missing_policy must be unavailable or fail")


@dataclass(frozen=True)
class LocalKnowledgeRootConfig:
    path: Path
    include: tuple[str, ...] = (
        "**/*.md",
        "**/*.txt",
        "**/*.json",
        "**/*.yaml",
        "**/*.yml",
        "**/*.csv",
    )
    exclude: tuple[str, ...] = (
        "**/~$*",
        "**/.git/**",
        "**/artifacts/**",
    )

    def __post_init__(self) -> None:
        raw_path = str(self.path)
        if "://" in raw_path or raw_path.casefold().startswith(("http:", "https:")):
            raise ValueError("Local knowledge roots must be filesystem paths, not URLs")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "include", tuple(str(item) for item in self.include))
        object.__setattr__(self, "exclude", tuple(str(item) for item in self.exclude))
        if not self.include:
            raise ValueError("Local knowledge root requires at least one include pattern")


@dataclass(frozen=True)
class LocalKnowledgeIngestionConfig:
    parser: str = "auto"
    rich_document_backend: str | None = None
    chunk_tokens: int = 480
    chunk_overlap: int = 64
    max_file_mb: int = 50
    follow_symlinks: bool = False
    extract_archives: bool = False

    def __post_init__(self) -> None:
        if self.parser != "auto":
            raise ValueError("Only the auto local-knowledge parser is currently supported")
        if self.chunk_tokens < 64:
            raise ValueError("local knowledge chunk_tokens must be at least 64")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("local knowledge chunk_overlap must be in [0, chunk_tokens)")
        if self.max_file_mb < 1:
            raise ValueError("local knowledge max_file_mb must be positive")
        if self.extract_archives:
            raise ValueError("Archive extraction is not supported by the offline knowledge loader")


@dataclass(frozen=True)
class LocalKnowledgeRetrievalConfig:
    mode: str = "lexical"
    lexical_backend: str = "sqlite_fts5"
    dense_enabled: bool = False
    embedding_model_path: Path | None = None
    allow_model_download: bool = False
    fusion: str = "rrf"
    top_k: int = 8
    lexical_candidates: int = 32
    dense_candidates: int = 32
    token_budget: int = 5000
    max_chunks_per_document: int = 3
    reranker_model_path: Path | None = None

    def __post_init__(self) -> None:
        for value in (self.embedding_model_path, self.reranker_model_path):
            if value is not None and (
                "://" in str(value)
                or str(value).casefold().startswith(("http:", "https:"))
            ):
                raise ValueError("Local knowledge model paths must not be URLs")
        if self.embedding_model_path is not None:
            object.__setattr__(self, "embedding_model_path", Path(self.embedding_model_path))
        if self.reranker_model_path is not None:
            object.__setattr__(self, "reranker_model_path", Path(self.reranker_model_path))
        if self.mode not in {"lexical", "dense", "hybrid"}:
            raise ValueError("local knowledge retrieval mode must be lexical, dense, or hybrid")
        if self.lexical_backend != "sqlite_fts5":
            raise ValueError("Only sqlite_fts5 lexical retrieval is supported")
        if self.fusion != "rrf":
            raise ValueError("Only reciprocal-rank fusion is supported")
        if self.allow_model_download:
            raise ValueError("Local knowledge models must not be downloaded at campaign runtime")
        if self.mode in {"dense", "hybrid"} and not self.dense_enabled:
            raise ValueError(f"retrieval mode {self.mode!r} requires dense_enabled=true")
        if self.dense_enabled and self.embedding_model_path is None:
            raise ValueError("dense local retrieval requires embedding_model_path")
        for name, value in (
            ("top_k", self.top_k),
            ("lexical_candidates", self.lexical_candidates),
            ("dense_candidates", self.dense_candidates),
            ("token_budget", self.token_budget),
            ("max_chunks_per_document", self.max_chunks_per_document),
        ):
            if value < 1:
                raise ValueError(f"local knowledge {name} must be positive")


@dataclass(frozen=True)
class LocalKnowledgeKGUpdateConfig:
    enabled: bool = True
    materialization: str = "retrieved_only"
    source_group: str = "local_documents"
    max_claims_per_round: int = 32
    contributes_to_selection: bool = False

    def __post_init__(self) -> None:
        if self.materialization != "retrieved_only":
            raise ValueError("Only retrieved_only local knowledge materialization is supported")
        if self.max_claims_per_round < 0:
            raise ValueError("max_claims_per_round must be non-negative")
        if self.contributes_to_selection:
            raise ValueError(
                "Local document evidence cannot contribute to selection before calibration"
            )


@dataclass(frozen=True)
class LeakageGuardConfig:
    enabled: bool = False
    mode: str = "generalize_and_filter"
    derive_protected_terms_from_task: bool = True
    protected_aliases: tuple[str, ...] = ()
    protected_accessions: tuple[str, ...] = ()
    strict_aliases_required: bool = True
    quarantine_target_documents: bool = True
    block_target_entities: bool = True
    minimum_sequence_fragment_length: int = 12

    def __post_init__(self) -> None:
        if self.mode != "generalize_and_filter":
            raise ValueError("Only generalize_and_filter leakage mode is supported")
        object.__setattr__(
            self, "protected_aliases", tuple(str(item) for item in self.protected_aliases)
        )
        object.__setattr__(
            self,
            "protected_accessions",
            tuple(str(item) for item in self.protected_accessions),
        )
        if self.minimum_sequence_fragment_length < 8:
            raise ValueError("minimum_sequence_fragment_length must be at least 8")


@dataclass
class LocalKnowledgeConfig:
    enabled: bool = False
    index_path: Path | None = None
    roots: tuple[LocalKnowledgeRootConfig | dict[str, Any], ...] = ()
    ingestion: LocalKnowledgeIngestionConfig | dict[str, Any] = field(
        default_factory=LocalKnowledgeIngestionConfig
    )
    retrieval: LocalKnowledgeRetrievalConfig | dict[str, Any] = field(
        default_factory=LocalKnowledgeRetrievalConfig
    )
    kg_update: LocalKnowledgeKGUpdateConfig | dict[str, Any] = field(
        default_factory=LocalKnowledgeKGUpdateConfig
    )
    leakage_guard: LeakageGuardConfig | dict[str, Any] = field(
        default_factory=LeakageGuardConfig
    )
    allow_remote_context: bool = False

    def __post_init__(self) -> None:
        if self.index_path is not None:
            if "://" in str(self.index_path) or str(self.index_path).casefold().startswith(
                ("http:", "https:")
            ):
                raise ValueError("Local knowledge index_path must be a filesystem path")
            self.index_path = Path(self.index_path)
        self.roots = tuple(
            item if isinstance(item, LocalKnowledgeRootConfig) else LocalKnowledgeRootConfig(**item)
            for item in self.roots
        )
        if not isinstance(self.ingestion, LocalKnowledgeIngestionConfig):
            self.ingestion = LocalKnowledgeIngestionConfig(**dict(self.ingestion))
        if not isinstance(self.retrieval, LocalKnowledgeRetrievalConfig):
            self.retrieval = LocalKnowledgeRetrievalConfig(**dict(self.retrieval))
        if not isinstance(self.kg_update, LocalKnowledgeKGUpdateConfig):
            self.kg_update = LocalKnowledgeKGUpdateConfig(**dict(self.kg_update))
        if not isinstance(self.leakage_guard, LeakageGuardConfig):
            self.leakage_guard = LeakageGuardConfig(**dict(self.leakage_guard))
        if self.enabled and not self.roots:
            raise ValueError("Enabled local knowledge requires at least one configured root")


@dataclass
class KnowledgeConfig:
    physchem: bool = True
    conservation: bool = True
    structure: bool = True
    kg: bool = True
    soft_weight: float = 0.20
    site_profiles: dict[int, dict[str, Any]] = field(default_factory=dict)
    legacy_mode: bool | None = None
    legacy_contributes_to_selection: bool = False
    fusion_mode: str = "independent_features"
    parameter_set_id: str = "knowledge-parameters:v1"
    providers: dict[str, KnowledgeProviderConfig | dict[str, Any]] = field(
        default_factory=dict
    )
    parameters: dict[str, LearnableParameterSpec | dict[str, Any]] = field(
        default_factory=dict
    )
    evidence_heartbeat_interval: int = 256
    local_knowledge: LocalKnowledgeConfig | dict[str, Any] = field(
        default_factory=LocalKnowledgeConfig
    )

    def __post_init__(self) -> None:
        if self.legacy_mode is None:
            self.legacy_mode = bool(self.site_profiles)
        if self.fusion_mode not in {"independent_features", "legacy_weighted_average"}:
            raise ValueError(f"Unsupported knowledge fusion_mode: {self.fusion_mode}")
        if self.evidence_heartbeat_interval < 1:
            raise ValueError("evidence_heartbeat_interval must be positive")
        if not isinstance(self.local_knowledge, LocalKnowledgeConfig):
            self.local_knowledge = LocalKnowledgeConfig(**dict(self.local_knowledge))
        self.providers = {
            str(name): (
                config
                if isinstance(config, KnowledgeProviderConfig)
                else KnowledgeProviderConfig(**dict(config))
            )
            for name, config in self.providers.items()
        }
        converted: dict[str, LearnableParameterSpec] = {}
        for name, spec in self.parameters.items():
            if isinstance(spec, LearnableParameterSpec):
                converted[str(name)] = spec
                continue
            raw = dict(spec)
            if raw.get("bounds") is not None:
                raw["bounds"] = tuple(float(item) for item in raw["bounds"])
            converted[str(name)] = LearnableParameterSpec(**raw)
        self.parameters = converted

    def provider(self, channel: str, default_kind: str) -> KnowledgeProviderConfig:
        configured = self.providers.get(channel)
        if configured is not None:
            return configured
        legacy_kind = {
            "physchem": "legacy_physchem",
            "conservation": "legacy_site_profile",
            "structure": "legacy_site_risk",
            "kg": "observation_association",
        }.get(channel, default_kind)
        return KnowledgeProviderConfig(
            kind=legacy_kind if self.legacy_mode else default_kind,
            enabled=bool(getattr(self, channel)),
            contributes_to_selection=(
                self.legacy_contributes_to_selection if self.legacy_mode else False
            ),
        )


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
    hypothesis_recency_decay: float = 1.0

    def __post_init__(self) -> None:
        if self.selection_driver not in {"auto", "agent_uq", "predictor", "random"}:
            raise ValueError("generation.selection_driver is invalid")
        if self.gp_length_scale <= 0 or self.gp_noise <= 0:
            raise ValueError("generation GP length scale and noise must be positive")
        if not 0 < self.hypothesis_recency_decay <= 1:
            raise ValueError("generation.hypothesis_recency_decay must be in (0, 1]")
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
        "query_physchem_delta",
        "query_evolutionary_profile",
        "query_structure_environment",
        "query_assay_association",
        "query_evidence_provenance",
        "query_local_knowledge",
        "query_structured_claims",
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
        knowledge_overrides = dict(ablation.get("knowledge", {}) or {})
        provider_overrides = dict(knowledge_overrides.pop("providers", {}) or {})
        if provider_overrides:
            merged_providers = dict(knowledge_raw.get("providers", {}) or {})
            merged_providers.update(provider_overrides)
            knowledge_raw["providers"] = merged_providers
        knowledge_raw.update(knowledge_overrides)

    task_values = dict(task_raw)
    for key in (
        "public_data_path",
        "oracle_data_path",
        "split_root",
        "reference_sequence_path",
    ):
        if task_values.get(key):
            path = Path(task_values[key])
            task_values[key] = path if path.is_absolute() else root / path
        else:
            task_values[key] = None
    structure_resources = []
    for raw_resource in task_values.get("structure_resources", ()) or ():
        resource = dict(raw_resource)
        if resource.get("path"):
            resource_path = Path(resource["path"])
            resource["path"] = (
                resource_path if resource_path.is_absolute() else root / resource_path
            )
        structure_resources.append(resource)
    task_values["structure_resources"] = tuple(structure_resources)
    task = TaskConfig(**task_values)
    profiles = {int(key): value for key, value in knowledge_raw.pop("site_profiles", {}).items()}
    provider_values: dict[str, Any] = {}
    for name, raw_provider in (knowledge_raw.pop("providers", {}) or {}).items():
        values = dict(raw_provider)
        if values.get("resource_path"):
            resource_path = Path(values["resource_path"])
            values["resource_path"] = (
                resource_path if resource_path.is_absolute() else root / resource_path
            )
        provider_values[str(name)] = KnowledgeProviderConfig(**values)
    knowledge_raw["providers"] = provider_values
    local_raw = dict(knowledge_raw.get("local_knowledge", {}) or {})
    if local_raw.get("index_path"):
        index_path = Path(local_raw["index_path"])
        local_raw["index_path"] = index_path if index_path.is_absolute() else root / index_path
    local_roots = []
    for raw_root in local_raw.get("roots", ()) or ():
        root_values = dict(raw_root)
        local_path = Path(root_values["path"])
        root_values["path"] = local_path if local_path.is_absolute() else root / local_path
        if "include" in root_values:
            root_values["include"] = tuple(str(item) for item in root_values["include"])
        if "exclude" in root_values:
            root_values["exclude"] = tuple(str(item) for item in root_values["exclude"])
        local_roots.append(LocalKnowledgeRootConfig(**root_values))
    local_raw["roots"] = tuple(local_roots)
    retrieval_values = dict(local_raw.get("retrieval", {}) or {})
    for key in ("embedding_model_path", "reranker_model_path"):
        if retrieval_values.get(key):
            model_path = Path(retrieval_values[key])
            retrieval_values[key] = model_path if model_path.is_absolute() else root / model_path
    if retrieval_values:
        local_raw["retrieval"] = LocalKnowledgeRetrievalConfig(**retrieval_values)
    if local_raw:
        knowledge_raw["local_knowledge"] = LocalKnowledgeConfig(**local_raw)
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
    removed = sorted(set(llm_raw).intersection(REMOVED_SDK_LLM_KEYS))
    if removed:
        raise ValueError(f"Removed Agents SDK settings are not supported: {removed}")
    if "runtime" in llm_raw:
        runtime = str(llm_raw.pop("runtime"))
        if runtime != "chat_completions":
            raise ValueError(f"Removed Agents SDK runtime is not supported: {runtime!r}")
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
