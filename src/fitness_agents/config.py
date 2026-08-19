from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from fitness_agents.contracts.capabilities import PredictorCapabilities

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
    initial_observations_path: Path | None = None
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
        uses_initial_only = self.initial_observations_path is not None
        if sum((uses_manifest, uses_legacy, uses_initial_only)) > 1:
            raise ValueError(
                "Task config cannot mix split_root, legacy public/oracle paths, and "
                "initial_observations_path"
            )
        if not uses_manifest and not uses_initial_only and not (
            self.public_data_path is not None and self.oracle_data_path is not None
        ):
            raise ValueError(
                "Task config requires split_root, initial_observations_path, or both "
                "public_data_path and oracle_data_path"
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
    capabilities: PredictorCapabilities | dict[str, bool] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.capabilities, dict):
            self.capabilities = PredictorCapabilities(**self.capabilities)


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
    a3m_path: Path | None = None
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
        if self.a3m_path is not None and self.kind != "msa_profile":
            raise ValueError("a3m_path is only valid for the msa_profile provider")
        if (
            self.a3m_path is not None
            and self.resource_path is not None
            and Path(self.a3m_path) != Path(self.resource_path)
        ):
            raise ValueError("Configure only one of a3m_path or resource_path for msa_profile")


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
    required_language: str | None = None

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
        if self.required_language is not None:
            language = str(self.required_language).strip().casefold()
            if language not in {"en"}:
                raise ValueError("local knowledge required_language currently supports only 'en'")
            object.__setattr__(self, "required_language", language)


def _validate_api_endpoint(endpoint: str, *, label: str) -> None:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"{label} must use HTTPS except for a local inference server")


def _validate_api_endpoint_path(endpoint: str, *, expected: str, label: str) -> None:
    path = urlparse(endpoint).path.rstrip("/")
    if path != expected:
        raise ValueError(f"{label} must use the full provider endpoint path {expected!r}")


@dataclass(frozen=True)
class EmbeddingAPIConfig:
    """Provider-neutral remote embedding configuration.

    ``provider`` selects the wire protocol; ``model_family`` records the model's
    semantic contract.  Keeping those fields separate lets BGE, E5, and open-weight
    Qwen models run behind any compatible TEI/OpenAI-style deployment.
    """

    provider: str
    endpoint: str
    api_key: str
    model: str
    model_family: str
    model_revision: str
    dimension: int
    max_input_tokens: int
    batch_size: int = 10
    timeout_seconds: float = 30.0
    max_retries: int = 3
    query_task: str | None = None
    document_task: str | None = None
    query_instruction: str | None = None
    document_instruction: str | None = None
    query_prefix: str = ""
    document_prefix: str = ""
    tokenizer_model_path: Path | None = None
    tokenizer_model_id: str | None = None
    tokenizer_revision: str | None = None
    normalize_embeddings: bool = True
    schema_version: str = "embedding-api:v1"

    def __post_init__(self) -> None:
        if self.schema_version != "embedding-api:v1":
            raise ValueError("Unsupported embedding API schema_version")
        if self.provider not in {"dashscope", "jina", "openai_compatible", "tei"}:
            raise ValueError(f"Unsupported embedding API provider: {self.provider}")
        if self.model_family not in {"qwen", "bge", "jina", "e5", "custom"}:
            raise ValueError(f"Unsupported embedding model_family: {self.model_family}")
        _validate_api_endpoint(self.endpoint, label="embedding API endpoint")
        if self.provider == "dashscope":
            _validate_api_endpoint_path(
                self.endpoint,
                expected="/api/v1/services/embeddings/text-embedding/text-embedding",
                label="DashScope text embedding endpoint",
            )
        if not self.api_key.strip():
            raise ValueError("embedding API api_key must not be empty")
        if not self.model.strip() or not self.model_revision.strip():
            raise ValueError("embedding API model and model_revision are required")
        if self.dimension < 1 or self.max_input_tokens < 1 or self.batch_size < 1:
            raise ValueError(
                "embedding API dimension, max_input_tokens, and batch_size must be positive"
            )
        if self.timeout_seconds <= 0 or not 0 <= self.max_retries <= 10:
            raise ValueError(
                "embedding API timeout must be positive and max_retries must be in [0, 10]"
            )
        if not self.normalize_embeddings:
            raise ValueError("Remote RAG embeddings must be normalized for cosine retrieval")
        if self.provider != "dashscope" and (
            self.query_instruction is not None or self.document_instruction is not None
        ):
            raise ValueError(
                "Only the DashScope embedding protocol accepts instruction fields; "
                "use provider task fields or explicit prefixes for other protocols"
            )
        if self.provider in {"openai_compatible", "tei"} and (
            self.query_task is not None or self.document_task is not None
        ):
            raise ValueError(
                "OpenAI-compatible/TEI embedding payloads do not accept task fields; "
                "use query_prefix/document_prefix"
            )
        if self.tokenizer_model_path is not None:
            raw_path = str(self.tokenizer_model_path)
            if "://" in raw_path:
                raise ValueError("embedding tokenizer_model_path must be a local path")
            object.__setattr__(self, "tokenizer_model_path", Path(self.tokenizer_model_path))


@dataclass(frozen=True)
class RerankerAPIConfig:
    provider: str
    endpoint: str
    api_key: str
    model: str
    model_family: str
    model_revision: str
    max_input_tokens: int
    max_documents: int = 64
    timeout_seconds: float = 30.0
    max_retries: int = 3
    instruction: str | None = None
    tokenizer_model_path: Path | None = None
    tokenizer_model_id: str | None = None
    tokenizer_revision: str | None = None
    score_kind: str = "probability"
    schema_version: str = "reranker-api:v1"

    def __post_init__(self) -> None:
        if self.schema_version != "reranker-api:v1":
            raise ValueError("Unsupported reranker API schema_version")
        if self.provider not in {"dashscope", "jina", "tei"}:
            raise ValueError(f"Unsupported reranker API provider: {self.provider}")
        if self.model_family not in {"qwen", "bge", "jina", "custom"}:
            raise ValueError(f"Unsupported reranker model_family: {self.model_family}")
        _validate_api_endpoint(self.endpoint, label="reranker API endpoint")
        if self.provider == "dashscope" and self.model == "qwen3-rerank":
            _validate_api_endpoint_path(
                self.endpoint,
                expected="/compatible-api/v1/reranks",
                label="DashScope qwen3-rerank endpoint",
            )
        if not self.api_key.strip():
            raise ValueError("reranker API api_key must not be empty")
        if not self.model.strip() or not self.model_revision.strip():
            raise ValueError("reranker API model and model_revision are required")
        if self.max_input_tokens < 1 or self.max_documents < 1:
            raise ValueError("reranker API max_input_tokens and max_documents must be positive")
        if self.timeout_seconds <= 0 or not 0 <= self.max_retries <= 10:
            raise ValueError(
                "reranker API timeout must be positive and max_retries must be in [0, 10]"
            )
        if self.score_kind not in {"probability", "raw_logit"}:
            raise ValueError("reranker API score_kind must be probability or raw_logit")
        if self.provider != "dashscope" and self.instruction is not None:
            raise ValueError("Only the DashScope reranker protocol accepts an instruction field")
        if self.tokenizer_model_path is not None:
            raw_path = str(self.tokenizer_model_path)
            if "://" in raw_path:
                raise ValueError("reranker tokenizer_model_path must be a local path")
            object.__setattr__(self, "tokenizer_model_path", Path(self.tokenizer_model_path))


@dataclass(frozen=True)
class LocalKnowledgeRetrievalConfig:
    mode: str = "lexical"
    lexical_backend: str = "sqlite_fts5"
    dense_enabled: bool = False
    embedding_backend: str = "local"
    embedding_model_path: Path | None = None
    embedding_api_config: EmbeddingAPIConfig | dict[str, Any] | None = None
    allow_model_download: bool = False
    fusion: str = "rrf"
    top_k: int = 8
    lexical_candidates: int = 32
    dense_candidates: int = 32
    token_budget: int = 5000
    max_chunks_per_document: int = 3
    reranker_model_path: Path | None = None
    reranker_backend: str = "auto"
    reranker_api_config: RerankerAPIConfig | dict[str, Any] | None = None
    reranker_model_id: str | None = None
    reranker_model_revision: str | None = None
    embedding_model_id: str | None = None
    embedding_model_revision: str | None = None
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""
    query_language: str = "en"
    strict_query_language: bool = False
    minimum_dense_similarity: float = 0.35
    require_dense_match_for_hybrid: bool = True
    minimum_reranker_score: float | None = None
    instruction_content_policy: str = "reject"
    dense_search_backend: str = "numpy_exact"
    max_exact_dense_chunks: int = 50000

    def __post_init__(self) -> None:
        if isinstance(self.embedding_api_config, dict):
            object.__setattr__(
                self,
                "embedding_api_config",
                EmbeddingAPIConfig(**dict(self.embedding_api_config)),
            )
        if isinstance(self.reranker_api_config, dict):
            object.__setattr__(
                self,
                "reranker_api_config",
                RerankerAPIConfig(**dict(self.reranker_api_config)),
            )
        for value in (self.embedding_model_path, self.reranker_model_path):
            if value is not None and (
                "://" in str(value) or str(value).casefold().startswith(("http:", "https:"))
            ):
                raise ValueError("Local knowledge model paths must not be URLs")
        if self.embedding_model_path is not None:
            object.__setattr__(self, "embedding_model_path", Path(self.embedding_model_path))
        if self.reranker_model_path is not None:
            object.__setattr__(self, "reranker_model_path", Path(self.reranker_model_path))
        if self.embedding_backend not in {"local", "api"}:
            raise ValueError("embedding_backend must be local or api")
        resolved_reranker_backend = self.reranker_backend
        if resolved_reranker_backend == "auto":
            if self.reranker_api_config is not None:
                resolved_reranker_backend = "api"
            elif self.reranker_model_path is not None:
                resolved_reranker_backend = "local"
            else:
                resolved_reranker_backend = "none"
            object.__setattr__(self, "reranker_backend", resolved_reranker_backend)
        if resolved_reranker_backend not in {"none", "local", "api"}:
            raise ValueError("reranker_backend must be none, local, api, or auto")
        if self.mode not in {"lexical", "dense", "hybrid"}:
            raise ValueError("local knowledge retrieval mode must be lexical, dense, or hybrid")
        if self.lexical_backend != "sqlite_fts5":
            raise ValueError("Only sqlite_fts5 lexical retrieval is supported")
        if self.fusion != "rrf":
            raise ValueError("Only reciprocal-rank fusion is supported")
        if self.query_language != "en":
            raise ValueError("local knowledge query_language currently supports only 'en'")
        if not -1.0 <= self.minimum_dense_similarity <= 1.0:
            raise ValueError("minimum_dense_similarity must be in [-1, 1]")
        if self.instruction_content_policy not in {"reject", "warn"}:
            raise ValueError("instruction_content_policy must be reject or warn")
        if self.dense_search_backend != "numpy_exact":
            raise ValueError("Only numpy_exact dense search is currently supported")
        if self.allow_model_download:
            raise ValueError("Local knowledge models must not be downloaded at campaign runtime")
        if self.mode in {"dense", "hybrid"} and not self.dense_enabled:
            raise ValueError(f"retrieval mode {self.mode!r} requires dense_enabled=true")
        if self.dense_enabled and self.embedding_backend == "local":
            if self.embedding_model_path is None:
                raise ValueError("dense local retrieval requires embedding_model_path")
            if self.embedding_api_config is not None:
                raise ValueError(
                    "local embedding_backend cannot also configure embedding_api_config"
                )
        if self.dense_enabled and self.embedding_backend == "api":
            if not isinstance(self.embedding_api_config, EmbeddingAPIConfig):
                raise ValueError("dense API retrieval requires embedding_api_config")
            if self.embedding_model_path is not None:
                raise ValueError("API embedding_backend cannot also configure embedding_model_path")
        if resolved_reranker_backend == "local":
            if self.reranker_model_path is None:
                raise ValueError("local reranker_backend requires reranker_model_path")
            if self.reranker_api_config is not None:
                raise ValueError("local reranker_backend cannot also configure reranker_api_config")
        if resolved_reranker_backend == "api":
            if not isinstance(self.reranker_api_config, RerankerAPIConfig):
                raise ValueError("API reranker_backend requires reranker_api_config")
            if self.reranker_model_path is not None:
                raise ValueError("API reranker_backend cannot also configure reranker_model_path")
        if resolved_reranker_backend == "none" and (
            self.reranker_model_path is not None or self.reranker_api_config is not None
        ):
            raise ValueError("reranker_backend=none cannot include a reranker configuration")
        for name, value in (
            ("top_k", self.top_k),
            ("lexical_candidates", self.lexical_candidates),
            ("dense_candidates", self.dense_candidates),
            ("token_budget", self.token_budget),
            ("max_chunks_per_document", self.max_chunks_per_document),
            ("max_exact_dense_chunks", self.max_exact_dense_chunks),
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
    selection_mode: str = "context_only"
    selection_calibration_path: Path | None = None

    def __post_init__(self) -> None:
        if self.materialization != "retrieved_only":
            raise ValueError("Only retrieved_only local knowledge materialization is supported")
        if self.max_claims_per_round < 0:
            raise ValueError("max_claims_per_round must be non-negative")
        if self.selection_mode not in {"context_only", "calibrated_candidate_projection"}:
            raise ValueError(
                "selection_mode must be context_only or calibrated_candidate_projection"
            )
        if self.selection_calibration_path is not None:
            object.__setattr__(
                self, "selection_calibration_path", Path(self.selection_calibration_path)
            )
        if self.contributes_to_selection and (
            self.selection_mode != "calibrated_candidate_projection"
            or self.selection_calibration_path is None
        ):
            raise ValueError(
                "Selection contribution requires calibrated_candidate_projection and "
                "selection_calibration_path"
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
    corpus_index_path: Path | None = None
    retrieval_overlay_path: Path | None = None
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
    leakage_guard: LeakageGuardConfig | dict[str, Any] = field(default_factory=LeakageGuardConfig)
    allow_remote_context: bool = False

    def __post_init__(self) -> None:
        for name in ("index_path", "corpus_index_path", "retrieval_overlay_path"):
            value = getattr(self, name)
            if value is None:
                continue
            if "://" in str(value) or str(value).casefold().startswith(("http:", "https:")):
                raise ValueError(f"Local knowledge {name} must be a filesystem path")
            setattr(self, name, Path(value))
        if self.corpus_index_path is None and self.index_path is not None:
            self.corpus_index_path = self.index_path
        if self.index_path is None and self.corpus_index_path is not None:
            self.index_path = self.corpus_index_path
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
    providers: dict[str, KnowledgeProviderConfig | dict[str, Any]] = field(default_factory=dict)
    parameters: dict[str, LearnableParameterSpec | dict[str, Any]] = field(default_factory=dict)
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


@dataclass(frozen=True)
class AgentQuotaAllocationConfig:
    """Fixed, auditable batch quotas for the one-hypothesis Agent-UQ path."""

    enabled: bool = False
    hypothesis_target: int = 8
    evidence_prior: int = 3
    coverage_exploration: int = 3
    matched_control: int = 2
    strong_hypothesis_threshold: float = 0.75

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.quotas().values()):
            raise ValueError("generation.quota_allocation quotas must be non-negative")
        if not 0.0 < self.strong_hypothesis_threshold <= 1.0:
            raise ValueError(
                "generation.quota_allocation.strong_hypothesis_threshold must be in (0, 1]"
            )

    def quotas(self) -> dict[str, int]:
        return {
            "hypothesis_target": self.hypothesis_target,
            "evidence_prior": self.evidence_prior,
            "coverage_exploration": self.coverage_exploration,
            "matched_control": self.matched_control,
        }

    @property
    def total(self) -> int:
        return sum(self.quotas().values())


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
    quota_allocation: AgentQuotaAllocationConfig | dict[str, Any] = field(
        default_factory=AgentQuotaAllocationConfig
    )

    def __post_init__(self) -> None:
        if isinstance(self.quota_allocation, dict):
            self.quota_allocation = AgentQuotaAllocationConfig(
                **dict(self.quota_allocation)
            )
        if self.selection_driver not in {
            "auto",
            "active_learning",
            "agent_uq",
            "predictor",
            "random",
        }:
            raise ValueError("generation.selection_driver is invalid")
        if self.gp_length_scale <= 0 or self.gp_noise <= 0:
            raise ValueError("generation GP length scale and noise must be positive")
        if not 0 < self.hypothesis_recency_decay <= 1:
            raise ValueError("generation.hypothesis_recency_decay must be in (0, 1]")
        if self.predictor_weight < 0:
            raise ValueError("generation.predictor_weight must be non-negative")


@dataclass(frozen=True)
class DesignerConfig:
    """Sequence-space definition for closed-pool or de novo mutation design.

    ``open_design`` is deliberately opt-in.  Its first implementation enumerates
    every configured single substitution and lets the posterior/acquisition layer
    rank the resulting full sequences; it never accepts a candidate pool as the
    proposal source.
    """

    space: str = "closed_pool"
    position_policy: str = "configured"
    proposer: str = "all_position_substitution"
    include_positions: tuple[int, ...] = ()
    exclude_positions: tuple[int, ...] = ()
    allowed_residues: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
    mutation_depth: int = 1
    max_preferred_positions: int = 12
    hypothesis_prior_weight: float = 0.20
    structure_constraint_weight: float = 0.20

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "include_positions", tuple(int(item) for item in self.include_positions)
        )
        object.__setattr__(
            self, "exclude_positions", tuple(int(item) for item in self.exclude_positions)
        )
        residues = tuple(str(item).strip().upper() for item in self.allowed_residues)
        object.__setattr__(self, "allowed_residues", residues)
        if self.space not in {"closed_pool", "open_design"}:
            raise ValueError("designer.space must be closed_pool or open_design")
        if self.position_policy not in {"configured", "all", "include", "all_except"}:
            raise ValueError("designer.position_policy is invalid")
        if self.space == "open_design" and self.proposer != "all_position_substitution":
            raise ValueError(
                "open_design currently requires proposer=all_position_substitution"
            )
        if self.position_policy == "include" and not self.include_positions:
            raise ValueError("designer.position_policy=include requires include_positions")
        if len(set(self.include_positions)) != len(self.include_positions):
            raise ValueError("designer.include_positions must be unique")
        if len(set(self.exclude_positions)) != len(self.exclude_positions):
            raise ValueError("designer.exclude_positions must be unique")
        canonical = frozenset("ACDEFGHIKLMNPQRSTVWY")
        if not residues or any(len(item) != 1 or item not in canonical for item in residues):
            raise ValueError("designer.allowed_residues must be canonical one-letter residues")
        if len(set(residues)) != len(residues):
            raise ValueError("designer.allowed_residues must be unique")
        if self.mutation_depth != 1:
            raise ValueError(
                "open_design MVP supports mutation_depth=1; multi-edit search requires "
                "combination-level posterior rescoring"
            )
        if self.max_preferred_positions < 1:
            raise ValueError("designer.max_preferred_positions must be positive")
        if self.hypothesis_prior_weight < 0 or self.structure_constraint_weight < 0:
            raise ValueError("designer soft-prior weights must be non-negative")


@dataclass(frozen=True)
class CalibratedPosteriorConfig:
    """Visible-label-only posterior configuration for active learning."""

    plugin: str = "visible_holdout_ensemble"
    predictor_models: tuple[ModelConfig, ...] = ()
    calibration_fraction: float = 0.20
    min_calibration_size: int = 8
    min_training_size: int = 8
    conformal_alpha: float = 0.10
    min_std: float = 1e-6
    variance_scale_bounds: tuple[float, float] = (0.25, 4.0)
    refit_full: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variance_scale_bounds",
            tuple(float(item) for item in self.variance_scale_bounds),
        )
        if not self.plugin:
            raise ValueError("active_learning.posterior.plugin must not be empty")
        if not 0 < self.calibration_fraction < 0.5:
            raise ValueError(
                "active_learning.posterior.calibration_fraction must be in (0, 0.5)"
            )
        if self.min_calibration_size < 2 or self.min_training_size < 4:
            raise ValueError(
                "active_learning posterior requires min_calibration_size >= 2 and "
                "min_training_size >= 4"
            )
        if not 0 < self.conformal_alpha < 1:
            raise ValueError("active_learning.posterior.conformal_alpha must be in (0, 1)")
        if self.min_std <= 0:
            raise ValueError("active_learning.posterior.min_std must be positive")
        if len(self.variance_scale_bounds) != 2:
            raise ValueError(
                "active_learning.posterior.variance_scale_bounds must contain two values"
            )
        lower, upper = self.variance_scale_bounds
        if lower <= 0 or lower > 1 or upper < 1 or lower >= upper:
            raise ValueError(
                "active_learning posterior variance_scale_bounds must straddle 1"
            )


@dataclass(frozen=True)
class HybridBatchAcquisitionConfig:
    """Quota-based hybrid acquisition over calibrated fitness predictions."""

    plugin: str = "hybrid_batch"
    exploitation_fraction: float = 0.50
    exploration_fraction: float = 0.25
    knowledge_fraction: float = 0.25
    ucb_beta: float = 1.0
    diversity_lambda: float = 0.10
    ood_penalty: float = 0.25
    knowledge_fitness_weight: float = 0.25
    validation_prior_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.plugin:
            raise ValueError("active_learning.acquisition.plugin must not be empty")
        fractions = (
            self.exploitation_fraction,
            self.exploration_fraction,
            self.knowledge_fraction,
        )
        if any(value < 0 for value in fractions) or not abs(sum(fractions) - 1.0) <= 1e-8:
            raise ValueError("active_learning acquisition fractions must be non-negative and sum to 1")
        if self.ucb_beta < 0:
            raise ValueError("active_learning.acquisition.ucb_beta must be non-negative")
        if self.diversity_lambda < 0 or self.ood_penalty < 0:
            raise ValueError(
                "active_learning acquisition diversity_lambda and ood_penalty must be non-negative"
            )
        if self.knowledge_fitness_weight < 0 or self.validation_prior_weight < 0:
            raise ValueError(
                "active_learning acquisition knowledge weights must be non-negative"
            )


@dataclass(frozen=True)
class ActiveLearningConfig:
    """Opt-in structured active-learning module configured independently of baselines."""

    enabled: bool = False
    module: str = "lightweight_calibrated_hybrid"
    posterior: CalibratedPosteriorConfig | dict[str, Any] = field(
        default_factory=CalibratedPosteriorConfig
    )
    acquisition: HybridBatchAcquisitionConfig | dict[str, Any] = field(
        default_factory=HybridBatchAcquisitionConfig
    )

    def __post_init__(self) -> None:
        if not self.module:
            raise ValueError("active_learning.module must not be empty")
        if isinstance(self.posterior, dict):
            object.__setattr__(
                self, "posterior", CalibratedPosteriorConfig(**dict(self.posterior))
            )
        if isinstance(self.acquisition, dict):
            object.__setattr__(
                self,
                "acquisition",
                HybridBatchAcquisitionConfig(**dict(self.acquisition)),
            )


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
        "query_feature_bundle",
        "query_kg_truncation_audit",
        "query_assay_association",
        "query_evidence_provenance",
        "query_local_knowledge",
        "query_structured_claims",
    )
    max_tool_calls: int = 3
    max_rows: int = 12
    use_counterevidence: bool = True
    stop_when_sufficient: bool = False
    feature_tool_strategy: str = "context_only"
    feature_channels: tuple[str, ...] = ("physchem", "conservation", "structure")
    feature_variant_limit: int = 1
    truncation_audit_enabled: bool = False
    truncation_audit_items: tuple[str, ...] = ()
    truncation_audit_sample_rows: int = 3

    def __post_init__(self) -> None:
        if self.max_tool_calls < 1 or self.max_rows < 1:
            raise ValueError("kg_interaction limits must be positive")
        allowed_strategies = {
            "context_only",
            "independent",
            "joint",
            "independent_and_joint",
        }
        if self.feature_tool_strategy not in allowed_strategies:
            raise ValueError(
                f"Unsupported feature_tool_strategy: {self.feature_tool_strategy!r}"
            )
        allowed_channels = {"physchem", "conservation", "structure"}
        unknown_channels = set(self.feature_channels).difference(allowed_channels)
        if unknown_channels or not self.feature_channels:
            raise ValueError(
                f"feature_channels must be a non-empty subset of {sorted(allowed_channels)}"
            )
        if len(self.feature_channels) != len(set(self.feature_channels)):
            raise ValueError("feature_channels must not contain duplicates")
        if self.feature_variant_limit < 1:
            raise ValueError("feature_variant_limit must be positive")
        independent_calls = (
            len(self.feature_channels) * self.feature_variant_limit
            if self.feature_tool_strategy in {"independent", "independent_and_joint"}
            else 0
        )
        joint_calls = (
            self.feature_variant_limit
            if self.feature_tool_strategy in {"joint", "independent_and_joint"}
            else 0
        )
        audit_calls = 1 if self.truncation_audit_enabled else 0
        minimum_calls = 1 + independent_calls + joint_calls + audit_calls
        if (
            self.feature_tool_strategy != "context_only" or self.truncation_audit_enabled
        ) and self.max_tool_calls < minimum_calls:
            raise ValueError(
                "max_tool_calls is too small for the configured feature/audit strategy; "
                f"requires at least {minimum_calls}"
            )
        required_operators = {"hypothesis_context"}
        channel_operators = {
            "physchem": "query_physchem_delta",
            "conservation": "query_evolutionary_profile",
            "structure": "query_structure_environment",
        }
        if self.feature_tool_strategy in {"independent", "independent_and_joint"}:
            required_operators.update(channel_operators[item] for item in self.feature_channels)
        if self.feature_tool_strategy in {"joint", "independent_and_joint"}:
            required_operators.add("query_feature_bundle")
        if self.truncation_audit_enabled:
            required_operators.add("query_kg_truncation_audit")
        missing_operators = required_operators.difference(self.enabled_operators)
        if (
            self.feature_tool_strategy != "context_only" or self.truncation_audit_enabled
        ) and missing_operators:
            raise ValueError(
                "feature/audit strategy operators are missing from enabled_operators: "
                f"{sorted(missing_operators)}"
            )
        if self.truncation_audit_enabled:
            if not self.truncation_audit_items:
                raise ValueError(
                    "truncation_audit_items must not be empty when the audit is enabled"
                )
            if len(self.truncation_audit_items) > self.max_rows:
                raise ValueError(
                    "truncation_audit_items cannot exceed max_rows because each item "
                    "produces one LLM-visible audit row"
                )
            if any(not item.strip() or len(item) > 128 for item in self.truncation_audit_items):
                raise ValueError("truncation audit items must contain 1 to 128 characters")
        if self.truncation_audit_sample_rows < 1:
            raise ValueError("truncation_audit_sample_rows must be positive")


@dataclass
class CriticConfig:
    enabled: bool = True
    mode: str = "rule"
    provider: str = "mock"
    model: str | None = None
    temperature: float = 0.0
    max_revision_attempts: int = 2
    # One runtime owns provider/output retries.  Scientific revisions above
    # are a separate budget and never multiply this value in CriticAgent.
    max_model_retries: int = 2
    max_output_retries: int = 1
    retry_backoff_seconds: float = 1.0
    request_timeout_seconds: float = 120.0
    max_input_chars: int = 80000
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
        if self.max_model_retries not in {0, 1, 2}:
            raise ValueError("critic.max_model_retries must be between 0 and 2")
        if self.max_output_retries not in {0, 1}:
            raise ValueError("critic.max_output_retries must be 0 or 1")
        if self.retry_backoff_seconds < 0:
            raise ValueError("critic.retry_backoff_seconds must be non-negative")
        if self.request_timeout_seconds <= 0:
            raise ValueError("critic.request_timeout_seconds must be positive")
        if self.max_input_chars < 4096:
            raise ValueError("critic.max_input_chars must be at least 4096")
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
    max_transport_retries: int = 2
    max_output_retries: int = 1
    retry_backoff_seconds: float = 1.0
    request_timeout_seconds: float = 120.0
    allow_unknown_evidence_stripping: bool = False
    max_input_chars: int = 80000

    def __post_init__(self) -> None:
        if self.max_transport_retries not in {0, 1, 2}:
            raise ValueError("llm.max_transport_retries must be between 0 and 2")
        if self.max_output_retries not in {0, 1}:
            raise ValueError("llm.max_output_retries must be 0 or 1")
        if self.retry_backoff_seconds < 0:
            raise ValueError("llm.retry_backoff_seconds must be non-negative")
        if self.request_timeout_seconds <= 0:
            raise ValueError("llm.request_timeout_seconds must be positive")
        if self.max_input_chars < 4096:
            raise ValueError("llm.max_input_chars must be at least 4096")


@dataclass
class HierarchicalHypothesisConfig:
    enabled: bool = False
    required_channels: tuple[str, ...] = ("physchem", "conservation", "structure")
    max_parallel_branches: int = 3
    max_child_revision_attempts: int = 1
    max_main_revision_attempts: int = 1
    formal_fail_closed: bool = True
    child_scientist_profiles: dict[str, str] = field(
        default_factory=lambda: {
            "physchem": "physchem_v1",
            "conservation": "conservation_v1",
            "structure": "structure_v1",
        }
    )
    child_critic_profiles: dict[str, str] = field(
        default_factory=lambda: {
            "physchem": "physchem_v1",
            "conservation": "conservation_v1",
            "structure": "structure_v1",
        }
    )
    main_scientist_profile: str = "synthesis_v1"
    main_critic_profile: str = "hypothesis_v1"
    child_max_tokens: int = 4096
    child_critic_max_tokens: int = 2048
    main_critic_max_tokens: int = 4096

    def __post_init__(self) -> None:
        allowed = {"physchem", "conservation", "structure"}
        if not self.required_channels or set(self.required_channels).difference(allowed):
            raise ValueError("hierarchical_hypothesis.required_channels is invalid")
        if self.max_parallel_branches not in {1, 2, 3}:
            raise ValueError("hierarchical_hypothesis.max_parallel_branches must be 1..3")
        if self.max_child_revision_attempts not in {0, 1, 2}:
            raise ValueError("max_child_revision_attempts must be between 0 and 2")
        if self.max_main_revision_attempts not in {0, 1, 2}:
            raise ValueError("max_main_revision_attempts must be between 0 and 2")
        for profiles in (self.child_scientist_profiles, self.child_critic_profiles):
            if set(profiles) != allowed or any(not value for value in profiles.values()):
                raise ValueError("hierarchical child profiles must cover all three channels")
        if min(
            self.child_max_tokens,
            self.child_critic_max_tokens,
            self.main_critic_max_tokens,
        ) < 512:
            raise ValueError("hierarchical role token budgets must be at least 512")


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
    designer: DesignerConfig = field(default_factory=DesignerConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    kg_interaction: KGInteractionRuntimeConfig = field(default_factory=KGInteractionRuntimeConfig)
    hierarchical_hypothesis: HierarchicalHypothesisConfig = field(
        default_factory=HierarchicalHypothesisConfig
    )
    llm_provider: str = "mock"
    knowledge_enabled: bool = False
    score_shuffle: bool = False
    evidence_deletion: bool = False
    run_label: str = ""
    condition: str = ""
    # Unused. Do not restore predictor-UCB evidence prefiltering; it is not
    # applied in the campaign loop and would change selection independently of Agent-UQ.
    evidence_prefilter_limit: int = 5000
    kg_ingest_evidence_limit: int = 120
    structured_kg_snapshot_mode: str = "live_only"

    def __post_init__(self) -> None:
        selected = self.generation.selection_driver == "active_learning"
        if selected != self.active_learning.enabled:
            raise ValueError(
                "generation.selection_driver=active_learning and active_learning.enabled=true "
                "must be configured together"
            )
        quota = self.generation.quota_allocation
        if quota.enabled:
            if self.generation.selection_driver != "agent_uq":
                raise ValueError(
                    "generation.quota_allocation requires selection_driver=agent_uq"
                )
            if quota.total != self.budget_per_round:
                raise ValueError(
                    "generation.quota_allocation quotas must sum to budget_per_round"
                )
        if self.kg_ingest_evidence_limit < 1:
            raise ValueError("kg_ingest_evidence_limit must be positive")
        if self.structured_kg_snapshot_mode not in {"live_only", "incremental_ids"}:
            raise ValueError(
                "structured_kg_snapshot_mode must be 'live_only' or 'incremental_ids'"
            )
        if self.designer.space == "open_design":
            if not (self.task.reference_sequence or self.task.reference_sequence_path):
                raise ValueError("open_design requires a complete task reference sequence")
            if self.designer.position_policy == "configured":
                raise ValueError(
                    "open_design requires an explicit position_policy; use 'all' to open "
                    "the complete reference sequence"
                )
            if self.generation.selection_driver != "active_learning":
                raise ValueError(
                    "open_design requires generation.selection_driver=active_learning so "
                    "posterior uncertainty participates in residue selection"
                )
            if self.candidate_limit > 0:
                raise ValueError(
                    "open_design does not use candidate_limit; set candidate_limit=0 so "
                    "the generated sequence space is not silently truncated"
                )
            # Local import avoids coupling configuration parsing to predictor backends.
            from fitness_agents.models.capabilities import predictor_capabilities

            posterior_models = (
                self.active_learning.posterior.predictor_models or (self.model,)
            )
            incompatible = [
                (
                    item.name,
                    item.feature_provider,
                    predictor_capabilities(item),
                )
                for item in posterior_models
                if not predictor_capabilities(item).supports_open_design
            ]
            if incompatible:
                details = ", ".join(
                    f"{name}/{provider} "
                    f"(supports_full_sequence={caps.supports_full_sequence}, "
                    f"supports_generated_sequences={caps.supports_generated_sequences})"
                    for name, provider, caps in incompatible
                )
                raise ValueError(
                    "open_design requires every posterior predictor to support full "
                    "generated sequences; incompatible predictors: "
                    f"{details}. GB1 four-site predictors cannot be used for open design."
                )
        elif self.task.initial_observations_path is not None:
            raise ValueError(
                "initial_observations_path is an open-design measurement source and cannot "
                "supply a closed_pool campaign"
            )


def _resolve_api_tokenizer_path(raw: dict[str, Any], root: Path) -> dict[str, Any]:
    values = dict(raw)
    if values.get("tokenizer_model_path"):
        tokenizer_path = Path(values["tokenizer_model_path"])
        values["tokenizer_model_path"] = (
            tokenizer_path if tokenizer_path.is_absolute() else root / tokenizer_path
        )
    return values


def load_inference_api_item(
    raw: dict[str, Any],
    *,
    expected_kind: str,
    root: Path,
) -> dict[str, Any]:
    """Resolve one typed item from a shared inference API catalog.

    The catalog owns workspace connection data and the secret reference. Each item owns its
    protocol, operation path, model, and model-specific limits. Item-local keys in ``raw`` are
    applied last so role settings such as an LLM profile can remain outside the shared catalog.
    """

    if not raw.get("api_catalog"):
        return dict(raw)
    item_name = str(raw.get("item", "")).strip()
    if not item_name:
        raise ValueError("api_catalog references require a non-empty item")
    catalog_path = Path(str(raw["api_catalog"]))
    catalog = read_yaml(catalog_path if catalog_path.is_absolute() else root / catalog_path, root)
    if catalog.get("schema_version") != "inference-api-catalog:v1":
        raise ValueError("Unsupported inference API catalog schema_version")
    connection = dict(catalog.get("connection", {}) or {})
    items = dict(catalog.get("items", {}) or {})
    if item_name not in items or not isinstance(items[item_name], dict):
        raise ValueError(f"Unknown inference API catalog item {item_name!r}")
    values = dict(items[item_name])
    kind = str(values.pop("kind", ""))
    if kind != expected_kind:
        raise ValueError(
            f"Inference API item {item_name!r} has kind {kind!r}, expected {expected_kind!r}"
        )
    origin = str(connection.get("origin", "")).rstrip("/")
    endpoint_path = values.pop("endpoint_path", None)
    base_url_path = values.pop("base_url_path", None)
    if endpoint_path is not None:
        if not origin:
            raise ValueError("Inference API catalog connection.origin is required")
        values["endpoint"] = f"{origin}/{str(endpoint_path).lstrip('/')}"
    if base_url_path is not None:
        if not origin:
            raise ValueError("Inference API catalog connection.origin is required")
        values["base_url"] = f"{origin}/{str(base_url_path).lstrip('/')}"
    values.setdefault("api_key", connection.get("api_key"))
    values.update(
        {
            key: value
            for key, value in raw.items()
            if key not in {"api_catalog", "item"}
        }
    )
    return values


def load_embedding_api_config(
    path: str | Path,
    *,
    root: Path | None = None,
) -> EmbeddingAPIConfig:
    base = root or project_root()
    raw = load_inference_api_item(
        read_yaml(path, base), expected_kind="embedding", root=base
    )
    return EmbeddingAPIConfig(**_resolve_api_tokenizer_path(raw, base))


def load_reranker_api_config(
    path: str | Path,
    *,
    root: Path | None = None,
) -> RerankerAPIConfig:
    base = root or project_root()
    raw = load_inference_api_item(
        read_yaml(path, base), expected_kind="reranker", root=base
    )
    return RerankerAPIConfig(**_resolve_api_tokenizer_path(raw, base))


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
        "initial_observations_path",
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
        for path_key in ("resource_path", "a3m_path"):
            if values.get(path_key):
                provider_path = Path(values[path_key])
                values[path_key] = (
                    provider_path if provider_path.is_absolute() else root / provider_path
                )
        provider_values[str(name)] = KnowledgeProviderConfig(**values)
    knowledge_raw["providers"] = provider_values
    local_raw = dict(knowledge_raw.get("local_knowledge", {}) or {})
    for key in ("index_path", "corpus_index_path", "retrieval_overlay_path"):
        if local_raw.get(key):
            local_path = Path(local_raw[key])
            local_raw[key] = local_path if local_path.is_absolute() else root / local_path
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
    embedding_api_value = retrieval_values.get("embedding_api_config")
    if isinstance(embedding_api_value, (str, Path)):
        retrieval_values["embedding_api_config"] = load_embedding_api_config(
            embedding_api_value, root=root
        )
    elif isinstance(embedding_api_value, dict):
        retrieval_values["embedding_api_config"] = EmbeddingAPIConfig(
            **_resolve_api_tokenizer_path(embedding_api_value, root)
        )
    reranker_api_value = retrieval_values.get("reranker_api_config")
    if isinstance(reranker_api_value, (str, Path)):
        retrieval_values["reranker_api_config"] = load_reranker_api_config(
            reranker_api_value, root=root
        )
    elif isinstance(reranker_api_value, dict):
        retrieval_values["reranker_api_config"] = RerankerAPIConfig(
            **_resolve_api_tokenizer_path(reranker_api_value, root)
        )
    if retrieval_values:
        local_raw["retrieval"] = LocalKnowledgeRetrievalConfig(**retrieval_values)
    kg_update_values = dict(local_raw.get("kg_update", {}) or {})
    if kg_update_values.get("selection_calibration_path"):
        calibration_path = Path(kg_update_values["selection_calibration_path"])
        kg_update_values["selection_calibration_path"] = (
            calibration_path if calibration_path.is_absolute() else root / calibration_path
        )
    if kg_update_values:
        local_raw["kg_update"] = LocalKnowledgeKGUpdateConfig(**kg_update_values)
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
    generation_raw["predictor_models"] = load_model_entries(generation_raw.get("predictor_models"))
    generation_raw["quota_allocation"] = _dataclass_from_mapping(
        AgentQuotaAllocationConfig,
        dict(generation_raw.get("quota_allocation", {}) or {}),
    )
    generation = _dataclass_from_mapping(GenerationConfig, generation_raw)
    designer_raw = dict(raw.get("designer", {}) or {})
    for key in ("include_positions", "exclude_positions", "allowed_residues"):
        if key in designer_raw:
            designer_raw[key] = tuple(designer_raw[key])
    designer = _dataclass_from_mapping(DesignerConfig, designer_raw)
    active_learning_raw = dict(raw.get("active_learning", {}) or {})
    posterior_raw = dict(active_learning_raw.get("posterior", {}) or {})
    posterior_raw["predictor_models"] = load_model_entries(
        posterior_raw.get("predictor_models")
    )
    active_learning_raw["posterior"] = _dataclass_from_mapping(
        CalibratedPosteriorConfig, posterior_raw
    )
    active_learning_raw["acquisition"] = _dataclass_from_mapping(
        HybridBatchAcquisitionConfig,
        dict(active_learning_raw.get("acquisition", {}) or {}),
    )
    active_learning = _dataclass_from_mapping(ActiveLearningConfig, active_learning_raw)
    validation_raw = dict(raw.get("validation", {}) or {})
    validation_raw["predictor_models"] = load_model_entries(validation_raw.get("predictor_models"))
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
    if "feature_channels" in interaction_raw:
        interaction_raw["feature_channels"] = tuple(
            str(item) for item in interaction_raw["feature_channels"]
        )
    if "truncation_audit_items" in interaction_raw:
        interaction_raw["truncation_audit_items"] = tuple(
            str(item) for item in interaction_raw["truncation_audit_items"]
        )
    kg_interaction = _dataclass_from_mapping(KGInteractionRuntimeConfig, interaction_raw)
    hierarchical_raw = dict(raw.get("hierarchical_hypothesis", {}) or {})
    if "required_channels" in hierarchical_raw:
        hierarchical_raw["required_channels"] = tuple(
            str(item) for item in hierarchical_raw["required_channels"]
        )
    hierarchical_hypothesis = _dataclass_from_mapping(
        HierarchicalHypothesisConfig, hierarchical_raw
    )
    critic_raw: dict[str, Any] = {}
    if raw.get("critic_config"):
        critic_raw.update(read_yaml(raw["critic_config"], root))
    if isinstance(raw.get("critic"), dict):
        critic_raw.update(raw["critic"])
    critic_raw = load_inference_api_item(critic_raw, expected_kind="llm", root=root)
    critic = _dataclass_from_mapping(CriticConfig, critic_raw)
    llm_raw: dict[str, Any] = {}
    if raw.get("llm_config"):
        llm_raw.update(read_yaml(raw["llm_config"], root))
    if isinstance(raw.get("llm"), dict):
        llm_raw.update(raw["llm"])
    if raw.get("llm_provider"):
        llm_raw["provider"] = raw["llm_provider"]
    llm_raw = load_inference_api_item(llm_raw, expected_kind="llm", root=root)
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
        designer=designer,
        active_learning=active_learning,
        validation=validation,
        evaluation=evaluation,
        output=output,
        kg_interaction=kg_interaction,
        hierarchical_hypothesis=hierarchical_hypothesis,
        output_root=root / raw.get("output_root", "artifacts/runs"),
        llm_provider=llm.provider,
        knowledge_enabled=bool(raw.get("knowledge_enabled", False)),
        score_shuffle=bool(raw.get("score_shuffle", False)),
        evidence_deletion=bool(raw.get("evidence_deletion", False)),
        run_label=str(raw.get("run_label", "")),
        condition=str(raw.get("condition") or raw.get("run_label") or raw["mode"]),
        evidence_prefilter_limit=int(raw.get("evidence_prefilter_limit", 5000)),
        kg_ingest_evidence_limit=int(raw.get("kg_ingest_evidence_limit", 120)),
        structured_kg_snapshot_mode=str(
            raw.get("structured_kg_snapshot_mode", "live_only")
        ),
    )
