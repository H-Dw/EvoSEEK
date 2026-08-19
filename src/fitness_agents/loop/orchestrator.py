from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import numpy as np

from fitness_agents.acquisition import create_policy
from fitness_agents.active_learning import create_active_learning_module
from fitness_agents.agents.client_registry import create_role_client_bundle
from fitness_agents.agents.critic import CriticAgent, OpenAICriticClient, RuleBasedCriticClient
from fitness_agents.agents.hypothesis_graph import HypothesisReviewGraph
from fitness_agents.agents.llm import create_llm_client
from fitness_agents.agents.main_hypothesis_critic import (
    RemoteMainHypothesisCritic,
    RuleBasedMainHypothesisCritic,
)
from fitness_agents.agents.output_guards import RevisionConstraints, critic_revision_payload
from fitness_agents.agents.rethink import create_rethink_client
from fitness_agents.agents.scientist import ScientistAgent
from fitness_agents.agents.subcritic import RemoteSubCritic, RuleBasedSubCritic
from fitness_agents.agents.subscientist import RemoteSubScientist, RuleBasedSubScientist
from fitness_agents.config import ExperimentConfig
from fitness_agents.contracts.agent_io import (
    ReThinkContextInput,
    RoleActivationState,
    ScientistContextInput,
)
from fitness_agents.contracts.hypothesis_pipeline import CompletionManifest
from fitness_agents.contracts.schemas import (
    CampaignPhase,
    CampaignState,
    DesignScore,
    Hypothesis,
    Prediction,
    SelectionRecord,
    ValidationRecord,
    Variant,
)
from fitness_agents.data import (
    load_campaign_fold_bundle,
    load_dataset_bundle,
    load_fold_bundle,
    load_fold_final_variants,
)
from fitness_agents.evaluation.hypotheses import (
    DeterministicHypothesisEvaluator,
    preregister_batch_median_test,
)
from fitness_agents.evaluation.metrics import loop_round_metrics, prediction_metrics
from fitness_agents.kg_interaction import (
    CompareVariantsOperator,
    EvidenceProvenanceOperator,
    ExplainVariantOperator,
    FeatureBundleOperator,
    FeatureEvidenceOperator,
    HypothesisContextOperator,
    InteractionAblationConfig,
    KGInteractionController,
    KGQueryContext,
    KGQueryPlan,
    KGQueryStep,
    KGTruncationAuditOperator,
    LocalKnowledgeQueryOperator,
    QueryIntent,
    StructuredClaimQueryOperator,
    runtime_truncation_audit_payload,
)
from fitness_agents.knowledge import KnowledgeEngine
from fitness_agents.models import create_predictor
from fitness_agents.mutation import (
    AgentQuotaBatchAcquisition,
    AgentUncertaintySelector,
    create_candidate_generator,
)
from fitness_agents.plugin_registry import PluginRegistry
from fitness_agents.protein_features import ProteinTaskContext
from fitness_agents.reporting import write_campaign_outputs
from fitness_agents.utils import JsonArtifactWriter, bind_progress, reset_progress, seed_everything
from fitness_agents.validation.batch import ApprovalGateway, BatchHardValidator, build_draft_batch

from .backends import ApprovalEnforcingBackend, CsvOracleBackend
from .review import BoundedReviewLoop, HypothesisRevisionRequested, ReviewRejected


def _descending_ranks(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda key: (values[key], key), reverse=True)
    return {variant_id: rank for rank, variant_id in enumerate(ordered, start=1)}


def _flatten_evidence(evidence: dict[str, list[Any]], limit: int = 120) -> list[Any]:
    entries = [item for bundle in evidence.values() for item in bundle]
    ranked = sorted(
        entries,
        key=lambda item: (
            item.quality_status != "unavailable",
            bool(item.contributes_to_selection),
            item.confidence * abs(item.score),
            item.evidence_id,
        ),
        reverse=True,
    )
    by_channel: dict[str, list[Any]] = {}
    for item in ranked:
        by_channel.setdefault(item.channel, []).append(item)
    balanced: list[Any] = []
    depth = 0
    channels = sorted(by_channel)
    while len(balanced) < limit and any(depth < len(by_channel[name]) for name in channels):
        for channel in channels:
            if depth < len(by_channel[channel]):
                balanced.append(by_channel[channel][depth])
                if len(balanced) == limit:
                    break
        depth += 1
    return balanced


def _decision_ingest_variants(
    observed_variants: Sequence[Any],
    *,
    selected_variants: Sequence[Any] = (),
    observations: Sequence[Any] = (),
    representative_limit: int = 2,
) -> list[Any]:
    """Variants allowed into the structured KG: observed, selected batch, and top visible reps."""

    observation_by_id = {item.variant_id: item for item in observations}
    ranked = sorted(
        observed_variants,
        key=lambda item: (
            observation_by_id[item.variant_id].fitness
            if item.variant_id in observation_by_id
            else float("-inf"),
            item.variant_id,
        ),
        reverse=True,
    )
    ordered: dict[str, Any] = {}
    for item in (*observed_variants, *selected_variants, *ranked[: max(representative_limit, 0)]):
        ordered[item.variant_id] = item
    return list(ordered.values())


def _shuffle_prediction_scores(
    predictions: Sequence[Prediction], rng: np.random.Generator
) -> list[Prediction]:
    means = np.asarray([prediction.fitness_mean for prediction in predictions])
    shuffled = rng.permutation(means)
    return [
        replace(prediction, fitness_mean=float(shuffled[index]))
        for index, prediction in enumerate(predictions)
    ]


def _validation_record_id(
    run_id: str,
    round_id: int,
    variant_id: str,
    validation_type: str,
    source_id: str,
) -> str:
    payload = f"{run_id}|{round_id}|{variant_id}|{validation_type}|{source_id}"
    return f"validation:{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


class CampaignRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        backend: Any | None = None,
        predictor_factory: Callable[..., Any] = create_predictor,
        agent: ScientistAgent | None = None,
        critic_agent: CriticAgent | None = None,
    ) -> None:
        self.config = config
        self.rng = seed_everything(config.seed)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        label = f"-{config.run_label}" if config.run_label else ""
        fold_label = f"-f{config.task.fold_index:02d}" if config.task.split_root else ""
        self.run_id = f"{config.mode}-s{config.seed}{fold_label}{label}-{timestamp}"
        self.writer = JsonArtifactWriter(config.output_root, self.run_id)
        self._split_root = config.task.split_root
        self._fold_index = config.task.fold_index
        if self._split_root is not None:
            fold_view = load_fold_bundle(self._split_root, self._fold_index, "agent")
            strategy = str(fold_view.manifest["strategy"])
            protocol_version = str(fold_view.manifest["protocol_version"])
            if (
                config.task.expected_split_strategy is not None
                and strategy != config.task.expected_split_strategy
            ):
                raise ValueError(
                    f"Configured strategy {config.task.expected_split_strategy!r} does not match "
                    f"manifest strategy {strategy!r}"
                )
            if (
                config.task.expected_protocol_version is not None
                and protocol_version != config.task.expected_protocol_version
            ):
                raise ValueError(
                    f"Configured protocol {config.task.expected_protocol_version!r} does not match "
                    f"manifest protocol {protocol_version!r}"
                )
            self.bundle = load_campaign_fold_bundle(self._split_root, self._fold_index)
            manifest_path = self._split_root / "manifest.public.json"
            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            assignment_sha256 = str(fold_view.fold_manifest["assignment_sha256"])
            if (
                config.task.expected_manifest_sha256 is not None
                and manifest_sha256 != config.task.expected_manifest_sha256
            ):
                raise ValueError("Manifest SHA-256 does not match the task configuration")
            if (
                config.task.expected_assignment_sha256 is not None
                and assignment_sha256 != config.task.expected_assignment_sha256
            ):
                raise ValueError("Fold assignment SHA-256 does not match the task configuration")
            self._data_source_record = {
                "kind": "manifest_fold",
                "split_root": str(self._split_root),
                "strategy": strategy,
                "protocol_version": protocol_version,
                "fold_index": self._fold_index,
                "manifest_sha256": manifest_sha256,
                "assignment_sha256": assignment_sha256,
            }
            role_counts = fold_view.fold_manifest["role_counts"]
            self._validation_count = int(role_counts.get("benchmark_validation", 0))
            self._final_test_count = int(role_counts.get("final_test", 0))
            default_backend = (
                CsvOracleBackend.from_fold(
                    self._split_root / f"fold_{self._fold_index:02d}",
                    query_budget=config.rounds * config.budget_per_round,
                )
                if backend is None
                else None
            )
        else:
            if config.task.public_data_path is None or config.task.oracle_data_path is None:
                raise ValueError("Legacy campaign requires public and oracle data paths")
            self.bundle = load_dataset_bundle(
                config.task.public_data_path, config.task.oracle_data_path
            )
            self._data_source_record = {
                "kind": "legacy_public_oracle",
                "public_data_path": str(config.task.public_data_path),
            }
            self._validation_count = len(self.bundle.validation_variants)
            self._final_test_count = len(self.bundle.final_test)
            default_backend = (
                CsvOracleBackend(config.task.oracle_data_path) if backend is None else None
            )
        raw_backend = backend if backend is not None else default_backend
        if raw_backend is None:
            raise AssertionError("Campaign backend initialization failed")
        self.approval_gateway = ApprovalGateway()
        self.backend = ApprovalEnforcingBackend(raw_backend, self.approval_gateway)
        self.predictor_factory = predictor_factory
        self.active_learning = (
            create_active_learning_module(
                config.active_learning,
                fallback_model=config.model,
                predictor_factory=predictor_factory,
                seed=config.seed,
            )
            if config.active_learning.enabled
            else None
        )
        self.task_context = ProteinTaskContext.from_task(config.task)
        self.knowledge = KnowledgeEngine(
            config.knowledge,
            graph_path=self.writer.run_dir / "knowledge_graph.sqlite",
            assay_id=config.task.assay_id,
            protein_id=config.task.protein_id,
            validation_config=config.validation,
            structured_graph_path=self.writer.run_dir / "structured_kg.sqlite",
            task_context=self.task_context,
            protein_name=config.task.protein_name,
            protein_aliases=config.task.protein_aliases,
            protein_accessions=config.task.protein_accessions,
            local_knowledge_enabled=config.knowledge_enabled,
            snapshot_mode=config.structured_kg_snapshot_mode,
        )
        self._scientist_local_context_allowed = bool(
            config.knowledge.local_knowledge.allow_remote_context or config.llm.provider == "mock"
        )
        self._critic_local_context_allowed = bool(
            config.knowledge.local_knowledge.allow_remote_context or config.critic.mode != "remote"
        )
        graph_tool = (
            self.knowledge.agent_tool()
            if config.knowledge_enabled and config.knowledge.kg and not config.evidence_deletion
            else None
        )
        llm_runtime_settings = {
            "model": self.config.llm.model,
            "base_url": self.config.llm.base_url,
            "api_key": self.config.llm.api_key,
            "temperature": self.config.llm.temperature,
            "max_tokens": self.config.llm.max_tokens,
            "reasoning_effort": self.config.llm.reasoning_effort,
            "thinking": self.config.llm.thinking,
            "max_transport_retries": self.config.llm.max_transport_retries,
            "max_output_retries": self.config.llm.max_output_retries,
            "retry_backoff_seconds": self.config.llm.retry_backoff_seconds,
            "request_timeout_seconds": self.config.llm.request_timeout_seconds,
            "allow_unknown_evidence_stripping": (
                self.config.llm.allow_unknown_evidence_stripping
            ),
            "max_input_chars": self.config.llm.max_input_chars,
        }
        role_clients = create_role_client_bundle(
            self.config.llm.provider,
            profile=self.config.llm.profile,
            **llm_runtime_settings,
        )
        main_scientist_client = role_clients.scientist
        if (
            config.hierarchical_hypothesis.enabled
            and config.llm.provider != "mock"
        ):
            main_scientist_client = create_llm_client(
                config.llm.provider,
                profile=config.hierarchical_hypothesis.main_scientist_profile,
                **{
                    **llm_runtime_settings,
                    "max_input_chars": (
                        config.hierarchical_hypothesis.main_max_input_chars
                        or config.llm.max_input_chars
                    ),
                },
            )
        self.agent = agent or ScientistAgent(
            main_scientist_client,
            task_context=self.task_context,
            objective=config.task.objective,
            knowledge_graph=graph_tool,
            design_space=config.designer.space,
            position_policy=config.designer.position_policy,
            max_preferred_positions=config.designer.max_preferred_positions,
        )
        self.kg_interaction = None
        if graph_tool is not None and config.kg_interaction.enabled:
            operators = PluginRegistry("query_operator")
            operators.register("hypothesis_context", HypothesisContextOperator(graph_tool))
            operators.register("explain_variant", ExplainVariantOperator(graph_tool))
            operators.register("compare_variants", CompareVariantsOperator(graph_tool))
            operators.register(
                "query_physchem_delta",
                FeatureEvidenceOperator("query_physchem_delta", "physchem", graph_tool),
            )
            operators.register(
                "query_evolutionary_profile",
                FeatureEvidenceOperator("query_evolutionary_profile", "conservation", graph_tool),
            )
            operators.register(
                "query_structure_environment",
                FeatureEvidenceOperator("query_structure_environment", "structure", graph_tool),
            )
            operators.register("query_feature_bundle", FeatureBundleOperator(graph_tool))
            operators.register(
                "query_kg_truncation_audit",
                KGTruncationAuditOperator(self.knowledge.structured_sink),
            )
            operators.register(
                "query_assay_association",
                FeatureEvidenceOperator("query_assay_association", "kg", graph_tool),
            )
            operators.register("query_evidence_provenance", EvidenceProvenanceOperator(graph_tool))
            if self.knowledge.local_knowledge is not None and self._scientist_local_context_allowed:
                operators.register(
                    "query_local_knowledge", LocalKnowledgeQueryOperator(self.knowledge)
                )
                operators.register(
                    "query_structured_claims", StructuredClaimQueryOperator(self.knowledge)
                )
            self.kg_interaction = KGInteractionController(
                operators,
                config=InteractionAblationConfig(
                    enabled_operators=frozenset(config.kg_interaction.enabled_operators),
                    max_tool_calls=config.kg_interaction.max_tool_calls,
                    use_counterevidence=config.kg_interaction.use_counterevidence,
                    stop_when_sufficient=config.kg_interaction.stop_when_sufficient,
                    read_only=True,
                ),
            )
        self.rethink_client = role_clients.rethink
        self.hypothesis_graph = None
        if config.hierarchical_hypothesis.enabled:
            channels = ("physchem", "conservation", "structure")
            if config.llm.provider == "mock":
                child_scientists = {channel: RuleBasedSubScientist() for channel in channels}
                child_critics = {channel: RuleBasedSubCritic() for channel in channels}
                main_hypothesis_critic = RuleBasedMainHypothesisCritic()
            else:
                child_scientists = {
                    channel: RemoteSubScientist(
                        profile=config.hierarchical_hypothesis.child_scientist_profiles[channel],
                        max_tokens=config.hierarchical_hypothesis.child_max_tokens,
                        max_input_chars=(
                            config.hierarchical_hypothesis.child_max_input_chars
                            or config.llm.max_input_chars
                        ),
                        **{
                            key: value
                            for key, value in llm_runtime_settings.items()
                            if key not in {"max_tokens", "max_input_chars"}
                        },
                        provider=config.llm.provider,
                    )
                    for channel in channels
                }
                child_critics = {
                    channel: RemoteSubCritic(
                        profile=config.hierarchical_hypothesis.child_critic_profiles[channel],
                        max_tokens=config.hierarchical_hypothesis.child_critic_max_tokens,
                        max_input_chars=(
                            config.hierarchical_hypothesis.critic_max_input_chars
                            or config.llm.max_input_chars
                        ),
                        thinking="disabled",
                        reasoning_effort=None,
                        **{
                            key: value
                            for key, value in llm_runtime_settings.items()
                            if key
                            not in {
                                "max_tokens",
                                "max_input_chars",
                                "thinking",
                                "reasoning_effort",
                            }
                        },
                        provider=config.llm.provider,
                    )
                    for channel in channels
                }
                main_hypothesis_critic = RemoteMainHypothesisCritic(
                    profile=config.hierarchical_hypothesis.main_critic_profile,
                    max_tokens=config.hierarchical_hypothesis.main_critic_max_tokens,
                    max_input_chars=(
                        config.hierarchical_hypothesis.critic_max_input_chars
                        or config.llm.max_input_chars
                    ),
                    **{
                        key: value
                        for key, value in llm_runtime_settings.items()
                        if key not in {"max_tokens", "max_input_chars"}
                    },
                    provider=config.llm.provider,
                )
            self.hypothesis_graph = HypothesisReviewGraph(
                child_scientists=child_scientists,
                child_critics=child_critics,
                main_critic=main_hypothesis_critic,
                required_channels=config.hierarchical_hypothesis.required_channels,
                max_parallel_branches=config.hierarchical_hypothesis.max_parallel_branches,
                max_child_revision_attempts=(
                    config.hierarchical_hypothesis.max_child_revision_attempts
                ),
                max_main_revision_attempts=(
                    config.hierarchical_hypothesis.max_main_revision_attempts
                ),
            )
        if critic_agent is None:
            rule_critic = RuleBasedCriticClient()
            if config.critic.mode == "remote" and config.critic.enabled:
                critic_client = OpenAICriticClient(
                    model=config.critic.model,
                    profile=config.critic.profile,
                    temperature=config.critic.temperature,
                    base_url=config.critic.base_url,
                    provider=config.critic.provider,
                    max_tokens=config.critic.max_tokens,
                    reasoning_effort=config.critic.reasoning_effort,
                    thinking=config.critic.thinking,
                    api_key=config.critic.api_key,
                    max_transport_retries=config.critic.max_model_retries,
                    max_output_retries=config.critic.max_output_retries,
                    retry_backoff_seconds=config.critic.retry_backoff_seconds,
                    request_timeout_seconds=config.critic.request_timeout_seconds,
                    max_input_chars=config.critic.max_input_chars,
                )
                critic_agent = CriticAgent(
                    critic_client,
                    # Provider/output retries are already owned by the client
                    # runtime.  Keeping this wrapper at zero prevents N x N
                    # request multiplication.
                    max_retries=0,
                    fallback=rule_critic if config.critic.fallback_policy == "rule" else None,
                )
            else:
                critic_agent = CriticAgent(rule_critic, max_retries=0)
        self.critic_agent = critic_agent
        self.hard_validator = BatchHardValidator(config.task, config.critic)
        self.review_loop = BoundedReviewLoop(
            validator=self.hard_validator,
            critic=self.critic_agent,
            max_revision_attempts=config.critic.max_revision_attempts,
            gateway=self.approval_gateway,
        )
        self.hypothesis_evaluator = DeterministicHypothesisEvaluator()
        self._progress_token = bind_progress(self.writer)
        self.generator = create_candidate_generator(
            config.mode,
            position_to_index=self.task_context.position_to_variant_index,
        )
        self.agent_selector = AgentUncertaintySelector(
            config.generation,
            position_to_index=self.task_context.position_to_variant_index,
        )
        self.agent_quota_acquisition = (
            AgentQuotaBatchAcquisition(config.generation.quota_allocation)
            if config.generation.quota_allocation.enabled
            else None
        )
        knowledge_weight = config.knowledge.soft_weight if config.knowledge_enabled else 0.0
        self.policy = create_policy(
            config.acquisition,
            beta=config.ucb_beta,
            knowledge_weight=knowledge_weight,
        )
        self.state = CampaignState(run_id=self.run_id, mode=config.mode, seed=config.seed)
        self._required_node_failures: list[str] = []
        self._fallback_nodes: list[str] = []
        self.validation_records: list[ValidationRecord] = []

    def _selection_driver(self) -> str:
        configured = self.config.generation.selection_driver
        if configured != "auto":
            return configured
        if self.config.mode == "fitness_direct":
            return "predictor"
        if self.config.mode == "random":
            return "random"
        return "agent_uq"

    def _fitness_predictors_used_for_generation(self, selection_driver: str) -> bool:
        if selection_driver == "agent_uq":
            return self.config.generation.use_fitness_predictors
        return selection_driver in {"active_learning", "predictor"}

    def _role_activation_state(
        self,
        role: str,
        *,
        evidence: Sequence[Any] = (),
        local_evidence: Sequence[Any] = (),
        interaction_result: Any | None = None,
    ) -> RoleActivationState:
        selection_driver = self._selection_driver()
        rag_configured = bool(
            self.config.knowledge_enabled and self.config.knowledge.local_knowledge.enabled
        )
        local_visible = bool(
            role == "scientist"
            and self._scientist_local_context_allowed
            or role == "critic"
            and self._critic_local_context_allowed
        )
        visible_evidence = [*evidence]
        if local_visible:
            visible_evidence.extend(local_evidence)
        available_channels = {
            item.channel
            for item in visible_evidence
            if getattr(item, "quality_status", None) != "unavailable"
        }
        unavailable_channels = {
            item.channel
            for item in visible_evidence
            if getattr(item, "quality_status", None) == "unavailable"
        }
        executed_tools = tuple(
            dict.fromkeys(pack.operator for pack in getattr(interaction_result, "packs", ()))
        )
        if (
            not executed_tools
            and getattr(self.agent, "knowledge_graph", None) is not None
            and self.config.mode in {"llm_agent", "knowledge_agent"}
        ):
            executed_tools = ("hypothesis_context",)
        configured_tools = (
            tuple(self.config.kg_interaction.enabled_operators)
            if self.kg_interaction is not None
            else (("hypothesis_context",) if getattr(self.agent, "knowledge_graph", None) else ())
        )
        rag_tools = {"query_local_knowledge", "query_structured_claims"}
        rag_tool_payload_present = any(
            pack.operator in rag_tools
            and bool(getattr(pack, "evidence", ()) or getattr(pack, "facts", ()))
            for pack in getattr(interaction_result, "packs", ())
        )
        rag_evidence_present = bool(
            local_visible
            and (
                any(
                    getattr(item, "channel", None) == "local_rag"
                    for item in visible_evidence
                )
                or rag_tool_payload_present
            )
        )
        return RoleActivationState(
            role=role,
            design_space="closed_pool",
            candidate_source="candidate_pool",
            candidate_pool_consulted=True,
            position_policy=self.config.designer.position_policy,
            selection_driver=selection_driver,
            active_learning_enabled=bool(
                selection_driver == "active_learning" and self.config.active_learning.enabled
            ),
            fitness_predictors_used_for_generation=(
                self._fitness_predictors_used_for_generation(selection_driver)
            ),
            rag_configured=rag_configured,
            rag_context_visible=bool(rag_configured and local_visible),
            rag_retrieval_performed=bool(self.knowledge.local_knowledge is not None),
            rag_evidence_present=rag_evidence_present,
            kg_configured=getattr(self.agent, "knowledge_graph", None) is not None,
            kg_interaction_enabled=self.kg_interaction is not None,
            configured_kg_tools=configured_tools,
            executed_kg_tools=executed_tools,
            kg_tool_results_present=bool(role == "scientist" and executed_tools),
            available_evidence_channels=tuple(sorted(available_channels)),
            unavailable_evidence_channels=tuple(sorted(unavailable_channels)),
        )

    def _create_and_fit_predictor(self, model_config: Any, observed_variants: list[Any], seed: int):
        predictor = self.predictor_factory(model_config, seed=seed)
        self._fit_predictor(predictor, observed_variants)
        return predictor

    def _run_kg_interaction(self, *, round_id: int, observed_variants: list[Any]):
        if self.kg_interaction is None:
            return None
        observation_by_id = {item.variant_id: item for item in self.state.observed}
        ranked = sorted(
            observed_variants,
            key=lambda item: (
                observation_by_id[item.variant_id].fitness,
                item.variant_id,
            ),
            reverse=True,
        )
        representative_ids = [item.variant_id for item in ranked[:2]]
        enabled_operators = frozenset(self.config.kg_interaction.enabled_operators)
        if "hypothesis_context" not in enabled_operators:
            raise ValueError("kg_interaction requires hypothesis_context in enabled_operators")
        steps = [
            KGQueryStep(
                "context",
                "hypothesis_context",
                QueryIntent.CONTEXT,
                {"limit": self.config.kg_interaction.max_rows},
            )
        ]
        if representative_ids and "query_assay_association" in enabled_operators:
            steps.append(
                KGQueryStep(
                    f"assay_{representative_ids[0]}",
                    "query_assay_association",
                    QueryIntent.EXPLAIN,
                    {"variant_id": representative_ids[0]},
                    ("context",),
                    "Retrieve observation-association KG evidence for the top visible variant.",
                )
            )
        channel_operator = {
            "physchem": "query_physchem_delta",
            "conservation": "query_evolutionary_profile",
            "structure": "query_structure_environment",
        }
        feature_variants = representative_ids[: self.config.kg_interaction.feature_variant_limit]
        if self.config.kg_interaction.feature_tool_strategy in {
            "independent",
            "independent_and_joint",
        }:
            for variant_id in feature_variants:
                for channel in self.config.kg_interaction.feature_channels:
                    operator = channel_operator[channel]
                    if operator not in enabled_operators:
                        continue
                    steps.append(
                        KGQueryStep(
                            f"feature_{channel}_{variant_id}",
                            operator,
                            QueryIntent.EXPLAIN,
                            {"variant_id": variant_id},
                            ("context",),
                            f"Retrieve the {channel} feature channel independently.",
                        )
                    )
        if self.config.kg_interaction.feature_tool_strategy in {
            "joint",
            "independent_and_joint",
        } and "query_feature_bundle" in enabled_operators:
            for variant_id in feature_variants:
                steps.append(
                    KGQueryStep(
                        f"feature_bundle_{variant_id}",
                        "query_feature_bundle",
                        QueryIntent.EXPLAIN,
                        {
                            "variant_id": variant_id,
                            "channels": list(self.config.kg_interaction.feature_channels),
                        },
                        ("context",),
                        "Retrieve the configured channels as one joint evidence bundle.",
                    )
                )
        rag_allowed = (
            self.knowledge.local_knowledge is not None and self._scientist_local_context_allowed
        )
        if rag_allowed and "query_local_knowledge" in enabled_operators:
            steps.append(
                KGQueryStep(
                    "local_knowledge",
                    "query_local_knowledge",
                    QueryIntent.SUPPORT,
                    {
                        "query": (
                            "general protein structure stability binding mutation "
                            "physicochemical epistasis knowledge"
                        ),
                        "anchors": [
                            "protein structure and stability",
                            "binding interface mutation effects",
                            "physicochemical substitution mechanisms",
                            "epistasis and residue interactions",
                        ],
                        "limit": self.config.kg_interaction.max_rows,
                    },
                    ("context",),
                )
            )
        if rag_allowed and "query_structured_claims" in enabled_operators:
            steps.append(
                KGQueryStep(
                    "structured_claims",
                    "query_structured_claims",
                    QueryIntent.SUPPORT,
                    {
                        "query": (
                            "binding affinity maturation library selection "
                            "mutation operational guideline"
                        ),
                        "limit": self.config.kg_interaction.max_rows,
                    },
                    ("context",),
                    "Read RAG-materialized claims from the structured KG snapshot.",
                )
            )
        if (
            self.config.kg_interaction.truncation_audit_enabled
            and "query_kg_truncation_audit" in enabled_operators
        ):
            steps.append(
                KGQueryStep(
                    "kg_truncation_audit",
                    "query_kg_truncation_audit",
                    QueryIntent.UNCERTAINTY,
                    {
                        "items": list(self.config.kg_interaction.truncation_audit_items),
                        "sample_rows": (self.config.kg_interaction.truncation_audit_sample_rows),
                    },
                    ("context",),
                    "Count keyword matches before max_rows and report missing bounded rows.",
                )
            )
        if representative_ids and "explain_variant" in enabled_operators:
            steps.append(
                KGQueryStep(
                    "explain",
                    "explain_variant",
                    QueryIntent.EXPLAIN,
                    {"variant_id": representative_ids[0]},
                    ("context",),
                )
            )
        if len(representative_ids) >= 2 and "compare_variants" in enabled_operators:
            steps.append(
                KGQueryStep(
                    "compare",
                    "compare_variants",
                    QueryIntent.COMPARE,
                    {"variant_ids": representative_ids},
                    ("context",),
                )
            )
        required_calls = len(steps)
        if required_calls > self.config.kg_interaction.max_tool_calls:
            raise ValueError(
                "kg_interaction.max_tool_calls is below the complete runtime plan; "
                f"configured={self.config.kg_interaction.max_tool_calls}, "
                f"required={required_calls}, steps={[item.step_id for item in steps]}"
            )
        return self.kg_interaction.execute(
            KGQueryPlan(
                plan_id=f"kgplan:{self.run_id}:r{round_id}",
                objective="Ground mutation hypotheses in visible evidence and counterevidence.",
                steps=tuple(steps),
                max_tool_calls=self.config.kg_interaction.max_tool_calls,
            ),
            KGQueryContext(
                run_id=self.run_id,
                round_id=round_id,
                allowed_variant_ids=frozenset(item.variant_id for item in observed_variants),
                max_rows=self.config.kg_interaction.max_rows,
            ),
        )

    def _record_kg_interaction(self, *, round_id: int, interaction_result: Any) -> None:
        self.writer.write_json(
            f"round_{round_id:02d}/kg_interaction.json",
            interaction_result,
        )
        truncation_audit = runtime_truncation_audit_payload(
            interaction_result,
            self.config.kg_interaction.truncation_audit_items,
        )
        if truncation_audit is not None:
            self.writer.write_json(
                f"round_{round_id:02d}/kg_truncation_audit.json",
                truncation_audit,
            )
        self.writer.event(
            "kg_interaction_completed",
            {
                "round_id": round_id,
                "operators": [pack.operator for pack in interaction_result.packs],
                "query_ids": [pack.query_id for pack in interaction_result.packs],
                "executed_steps": interaction_result.executed_steps,
                "skipped_steps": interaction_result.skipped_steps,
                "stop_reason": interaction_result.stop_reason,
                "feature_tool_strategy": self.config.kg_interaction.feature_tool_strategy,
                "feature_channels": self.config.kg_interaction.feature_channels,
                "kg_truncation_audit": (
                    {
                        "any_truncated": truncation_audit.get("any_truncated"),
                        "missing_items": truncation_audit.get("missing_items"),
                    }
                    if truncation_audit is not None
                    else None
                ),
            },
        )

    def _evidence_ingest_limit(self) -> int:
        return max(1, int(self.config.kg_ingest_evidence_limit))

    def _flatten_round_evidence(self, evidence: dict[str, list[Any]]) -> list[Any]:
        return _flatten_evidence(evidence, limit=self._evidence_ingest_limit())

    def _scientist_prompt_evidence(self, evidence: dict[str, list[Any]]) -> list[Any]:
        return _flatten_evidence(
            evidence, limit=max(1, int(self.config.scientist_prompt_evidence_limit))
        )

    def _persist_evidence_map(
        self,
        variants: Sequence[Any],
        evidence: dict[str, list[Any]],
        *,
        variant_ids: set[str] | None = None,
    ) -> None:
        if not evidence:
            return
        selected = [
            item for item in variants if variant_ids is None or item.variant_id in variant_ids
        ]
        if selected:
            self.knowledge.graph.add_variants(selected)
        items = [
            item
            for variant_id, bundle in evidence.items()
            if variant_ids is None or variant_id in variant_ids
            for item in bundle
        ]
        if items:
            self.knowledge.graph.add_evidence(items)

    def _config_record(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "condition": self.config.condition or self.config.mode,
            "run_label": self.config.run_label,
            "seed": self.config.seed,
            "rounds": self.config.rounds,
            "budget_per_round": self.config.budget_per_round,
            "candidate_limit": self.config.candidate_limit,
            "acquisition": self.config.acquisition,
            "knowledge_enabled": self.config.knowledge_enabled,
            "knowledge_channels": {
                "physchem": self.config.knowledge.physchem,
                "conservation": self.config.knowledge.conservation,
                "structure": self.config.knowledge.structure,
                "kg": self.config.knowledge.kg,
            },
            "protein_context": {
                "context_id": self.task_context.context_id,
                "protein_id": self.task_context.protein_id,
                "sequence_mode": self.task_context.sequence_mode,
                "numbering_scheme": self.task_context.numbering_scheme,
                "mutable_positions": list(self.task_context.mutable_positions),
                "wild_type_sites": self.task_context.wild_type_code,
                "structure_resource_ids": [
                    item.resource_id for item in self.task_context.structure_resources
                ],
            },
            "knowledge_runtime": {
                "fusion_mode": self.config.knowledge.fusion_mode,
                "legacy_mode": self.config.knowledge.legacy_mode,
                "parameter_set_id": self.config.knowledge.parameter_set_id,
                "provider_status": self.knowledge.provider_status,
                "providers": self.knowledge.provider_configs,
                "parameters": self.config.knowledge.parameters,
                "local_knowledge": {
                    "enabled": self.knowledge.local_knowledge is not None,
                    "index_path": (
                        str(self.config.knowledge.local_knowledge.corpus_index_path)
                        if self.config.knowledge.local_knowledge.corpus_index_path is not None
                        else None
                    ),
                    "retrieval_overlay_path": (
                        str(self.config.knowledge.local_knowledge.retrieval_overlay_path)
                        if self.config.knowledge.local_knowledge.retrieval_overlay_path is not None
                        else None
                    ),
                    "retrieval_mode": self.config.knowledge.local_knowledge.retrieval.mode,
                    "dense_enabled": (
                        self.config.knowledge.local_knowledge.retrieval.dense_enabled
                    ),
                    "embedding_model_id": (
                        self.config.knowledge.local_knowledge.retrieval.embedding_model_id
                    ),
                    "embedding_model_revision": (
                        self.config.knowledge.local_knowledge.retrieval.embedding_model_revision
                    ),
                    "embedding_fingerprint": (
                        self.knowledge.local_knowledge.embedding_backend.fingerprint
                        if self.knowledge.local_knowledge is not None
                        and self.knowledge.local_knowledge.embedding_backend is not None
                        else None
                    ),
                    "reranker_fingerprint": (
                        self.knowledge.local_knowledge.reranker_backend.fingerprint
                        if self.knowledge.local_knowledge is not None
                        and self.knowledge.local_knowledge.reranker_backend is not None
                        else None
                    ),
                    "leakage_guard_enabled": (
                        self.config.knowledge.local_knowledge.leakage_guard.enabled
                    ),
                    "allow_remote_context": (
                        self.config.knowledge.local_knowledge.allow_remote_context
                    ),
                    "contributes_to_selection": (
                        self.config.knowledge.local_knowledge.kg_update.contributes_to_selection
                    ),
                    "selection_mode": (
                        self.config.knowledge.local_knowledge.kg_update.selection_mode
                    ),
                    "selection_calibration_sha256": (
                        self.knowledge.local_knowledge.selection_projector.sha256
                        if self.knowledge.local_knowledge is not None
                        and self.knowledge.local_knowledge.selection_projector is not None
                        else None
                    ),
                    "scientist_context_allowed": self._scientist_local_context_allowed,
                    "critic_context_allowed": self._critic_local_context_allowed,
                    "build_report": self.knowledge.local_knowledge_build_report,
                },
            },
            "llm_provider": self.config.llm.provider,
            "llm": {
                "provider": self.config.llm.provider,
                "runtime": "native_chat_completions",
                "profile": self.config.llm.profile,
                "profile_sha256": getattr(self.agent.client, "profile_sha256", None),
                "model": self.config.llm.model,
                "base_url": self.config.llm.base_url,
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
                "reasoning_effort": self.config.llm.reasoning_effort,
                "thinking": self.config.llm.thinking,
                "max_transport_retries": self.config.llm.max_transport_retries,
                "max_output_retries": self.config.llm.max_output_retries,
                "retry_backoff_seconds": self.config.llm.retry_backoff_seconds,
                "request_timeout_seconds": self.config.llm.request_timeout_seconds,
                "allow_unknown_evidence_stripping": (
                    self.config.llm.allow_unknown_evidence_stripping
                ),
                "max_input_chars": self.config.llm.max_input_chars,
                "trace_role": "observability_only",
                "scientific_state_source": "wet_dry_kg_artifact",
                "api_key_ref": self.config.llm.api_key
                if isinstance(self.config.llm.api_key, str)
                and self.config.llm.api_key.startswith("env:")
                else ("configured" if self.config.llm.api_key else None),
            },
            "critic": {
                "enabled": self.config.critic.enabled,
                "mode": self.config.critic.mode,
                "provider": self.config.critic.provider,
                "model": self.config.critic.model,
                "profile": self.config.critic.profile,
                "base_url": self.config.critic.base_url,
                "max_revision_attempts": self.config.critic.max_revision_attempts,
                "max_transport_retries": self.config.critic.max_model_retries,
                "max_output_retries": self.config.critic.max_output_retries,
                "fallback_policy": self.config.critic.fallback_policy,
                "api_key_ref": self.config.critic.api_key
                if isinstance(self.config.critic.api_key, str)
                and self.config.critic.api_key.startswith("env:")
                else ("configured" if self.config.critic.api_key else None),
            },
            "score_shuffle": self.config.score_shuffle,
            "evidence_deletion": self.config.evidence_deletion,
            "evidence_prefilter_limit": self.config.evidence_prefilter_limit,
            "kg_ingest_evidence_limit": self.config.kg_ingest_evidence_limit,
            "scientist_prompt_evidence_limit": self.config.scientist_prompt_evidence_limit,
            "structured_kg_snapshot_mode": self.config.structured_kg_snapshot_mode,
            "generation": {
                "selection_driver": self._selection_driver(),
                "use_fitness_predictors": self.config.generation.use_fitness_predictors,
                "predictor_models": [item.name for item in self.config.generation.predictor_models],
                "hypothesis_weight": self.config.generation.hypothesis_weight,
                "evidence_weight": self.config.generation.evidence_weight,
                "prior_weight": self.config.generation.prior_weight,
                "uncertainty_beta": self.config.generation.uncertainty_beta,
                "predictor_weight": self.config.generation.predictor_weight,
                "gp_length_scale": self.config.generation.gp_length_scale,
                "hypothesis_recency_decay": (self.config.generation.hypothesis_recency_decay),
                "quota_allocation": {
                    "enabled": self.config.generation.quota_allocation.enabled,
                    "plugin": (
                        self.agent_quota_acquisition.name
                        if self.agent_quota_acquisition is not None
                        else None
                    ),
                    "quotas": self.config.generation.quota_allocation.quotas(),
                    "strong_hypothesis_threshold": (
                        self.config.generation.quota_allocation.strong_hypothesis_threshold
                    ),
                },
            },
            "active_learning": {
                "enabled": self.config.active_learning.enabled,
                "module": self.config.active_learning.module,
                "posterior_plugin": self.config.active_learning.posterior.plugin,
                "posterior_models": [
                    item.name
                    for item in (
                        self.config.active_learning.posterior.predictor_models
                        or (self.config.model,)
                    )
                ],
                "calibration_fraction": (
                    self.config.active_learning.posterior.calibration_fraction
                ),
                "min_calibration_size": (
                    self.config.active_learning.posterior.min_calibration_size
                ),
                "acquisition_plugin": self.config.active_learning.acquisition.plugin,
                "fractions": {
                    "exploitation": (self.config.active_learning.acquisition.exploitation_fraction),
                    "exploration": (self.config.active_learning.acquisition.exploration_fraction),
                    "knowledge": self.config.active_learning.acquisition.knowledge_fraction,
                },
            },
            "validation": {
                "enabled": self.config.validation.enabled,
                "primary_model": self.config.model.name,
                "additional_models": [
                    item.name for item in self.config.validation.predictor_models
                ],
                "wet_weight": self.config.validation.wet_weight,
                "dry_weight_cap": self.config.validation.dry_weight_cap,
                "recency_decay": self.config.validation.recency_decay,
                "rethink_enabled": self.config.validation.rethink_enabled,
            },
            "kg_interaction": {
                "enabled": self.kg_interaction is not None,
                "enabled_operators": list(self.config.kg_interaction.enabled_operators),
                "feature_tool_strategy": self.config.kg_interaction.feature_tool_strategy,
                "feature_channels": list(self.config.kg_interaction.feature_channels),
                "feature_variant_limit": self.config.kg_interaction.feature_variant_limit,
                "truncation_audit_enabled": (self.config.kg_interaction.truncation_audit_enabled),
                "truncation_audit_items": list(self.config.kg_interaction.truncation_audit_items),
                "truncation_audit_sample_rows": (
                    self.config.kg_interaction.truncation_audit_sample_rows
                ),
                "max_tool_calls": self.config.kg_interaction.max_tool_calls,
                "max_rows": self.config.kg_interaction.max_rows,
                "stop_when_sufficient": self.config.kg_interaction.stop_when_sufficient,
            },
            "hierarchical_hypothesis": {
                "enabled": self.config.hierarchical_hypothesis.enabled,
                "required_channels": list(
                    self.config.hierarchical_hypothesis.required_channels
                ),
                "max_parallel_branches": (
                    self.config.hierarchical_hypothesis.max_parallel_branches
                ),
                "max_child_revision_attempts": (
                    self.config.hierarchical_hypothesis.max_child_revision_attempts
                ),
                "max_main_revision_attempts": (
                    self.config.hierarchical_hypothesis.max_main_revision_attempts
                ),
                "formal_fail_closed": (
                    self.config.hierarchical_hypothesis.formal_fail_closed
                ),
                "main_scientist_profile": (
                    self.config.hierarchical_hypothesis.main_scientist_profile
                ),
                "main_critic_profile": (
                    self.config.hierarchical_hypothesis.main_critic_profile
                ),
                "main_max_input_chars": (
                    self.config.hierarchical_hypothesis.main_max_input_chars
                    or self.config.llm.max_input_chars
                ),
                "child_max_input_chars": (
                    self.config.hierarchical_hypothesis.child_max_input_chars
                    or self.config.llm.max_input_chars
                ),
                "critic_max_input_chars": (
                    self.config.hierarchical_hypothesis.critic_max_input_chars
                    or self.config.llm.max_input_chars
                ),
            },
            "evaluation": {
                "metrics": list(self.config.evaluation.metrics),
                "top_k": self.config.evaluation.top_k,
            },
            "output": {
                "artifacts": list(self.config.output.artifacts),
                "top_k": self.config.output.top_k,
            },
            "model": self.config.model.name,
            "model_device": self.config.model.device,
            "model_allow_device_fallback": self.config.model.allow_device_fallback,
            "model_batch_size": self.config.model.batch_size,
            "model_backend_factory": self.config.model.backend_factory,
            "model_checkpoint": self.config.model.checkpoint,
            "model_options": self.config.model.options,
            "data_source": self._data_source_record,
            "fitness_semantics": {
                "scale": self.config.task.fitness_scale,
                "transform": self.config.task.fitness_transform,
            },
        }

    def _repropose_after_critic(
        self,
        decision: Any,
        rejected: Any,
        *,
        observed_variants: Sequence[Variant],
        evidence: dict[str, list[Any]],
        local_evidence: Sequence[Any],
        interaction_result: Any | None,
    ) -> Any:
        revision = critic_revision_payload(decision=decision, rejected_hypothesis=rejected)
        evidence_list = [
            *(local_evidence if self._scientist_local_context_allowed else ()),
            *self._scientist_prompt_evidence(evidence),
        ]
        activation_state = self._role_activation_state(
            "scientist",
            evidence=self._scientist_prompt_evidence(evidence),
            local_evidence=local_evidence,
            interaction_result=interaction_result,
        )
        hypothesis = self.agent.propose_hypothesis(
            self.state,
            observed_variants,
            self.state.observed,
            evidence_list,
            kg_interaction=interaction_result,
            activation_state=activation_state,
            critic_revision=revision,
            hypothesis_attempt=1,
        )
        if hypothesis.preferred_residues == rejected.preferred_residues:
            retried = self.agent.propose_hypothesis(
                self.state,
                observed_variants,
                self.state.observed,
                evidence_list,
                kg_interaction=interaction_result,
                activation_state=activation_state,
                critic_revision={**revision, "identical_residues_rejected": True},
                hypothesis_attempt=2,
            )
            if (
                retried.preferred_residues != rejected.preferred_residues
                or retried.statement != rejected.statement
            ):
                hypothesis = retried
            else:
                self.writer.event(
                    "llm_output_warning",
                    {
                        "message": "Revised hypothesis kept the same preferred_residues",
                        "hypothesis_id": hypothesis.hypothesis_id,
                    },
                )
        self.state.hypotheses.append(hypothesis)
        self.knowledge.graph.add_hypothesis(
            hypothesis.hypothesis_id,
            self.state.round_id,
            hypothesis.statement,
            hypothesis.evidence_ids,
        )
        self.writer.event("hypothesis_proposed", hypothesis.__dict__)
        return hypothesis

    def _progress(
        self,
        event_type: str | None,
        message: str,
        *,
        phase: CampaignPhase | None = None,
        persist: bool | None = None,
        **payload: Any,
    ) -> None:
        if phase is not None:
            self.state.phase = phase
        persist = persist if persist is not None else event_type is not None
        self.writer.report(
            event_type,
            message=message,
            persist=persist,
            round_id=self.state.round_id,
            phase=self.state.phase.value,
            **payload,
        )

    def _final_test_variants(self) -> list[Any]:
        if self._split_root is not None:
            return load_fold_final_variants(self._split_root, self._fold_index)
        return list(self.bundle.final_test)

    def _fit_predictor(
        self,
        predictor: Any,
        observed_variants: list[Any],
    ) -> None:
        if self._split_root is not None:
            predictor.fit(observed_variants, self.state.observed)
            return
        predictor.fit(
            observed_variants,
            self.state.observed,
            self.bundle.validation_variants,
            self.bundle.validation_observations,
        )

    def run(self) -> dict[str, Any]:
        try:
            return self._run_campaign()
        except Exception:
            with contextlib.suppress(Exception):
                self._progress("campaign_failed", "campaign failed")
            raise
        finally:
            reset_progress(self._progress_token)

    def _run_campaign(self) -> dict[str, Any]:
        public_by_id = {
            variant.variant_id: variant
            for variant in (self.bundle.initial_variants + self.bundle.oracle_pool)
        }
        observed_variants = list(self.bundle.initial_variants)
        self.state.observed = list(self.bundle.initial_observations)
        self.state.revealed_variant_ids = {item.variant_id for item in self.state.observed}
        self.knowledge.update(observed_variants, self.state.observed)
        remaining = list(self.bundle.oracle_pool)
        selection_driver = self._selection_driver()
        wild_type = next(
            (item for item in observed_variants if item.mutation_count == 0),
            None,
        )
        if wild_type is None:
            wild_type = Variant(
                variant_id=f"configured-wt:{self.config.task.protein_id}",
                variant=self.config.task.wild_type_sites,
                sequence=self.config.task.wild_type_sites,
                mutation_notation="WT",
                mutation_count=0,
                split_role="configured_reference",
            )
        round_metrics: list[dict[str, float]] = []
        rounds_aborted = 0
        self.writer.write_json("config.json", self._config_record())
        self.writer.write_json(
            "knowledge/manifest.json",
            {
                "protein_context": self._config_record()["protein_context"],
                **self._config_record()["knowledge_runtime"],
                "scientific_state_source": "wet_dry_kg_artifact",
                "phase6_status": "future_todo_expensive_adapters_disabled",
            },
        )
        self.writer.event(
            "campaign_started",
            {
                "run_id": self.run_id,
                "initial_count": len(observed_variants),
                "validation_count": self._validation_count,
                "oracle_pool_count": len(remaining),
                "final_test_count": self._final_test_count,
                "data_source": self._data_source_record,
            },
        )
        self._progress(
            None,
            (
                f"campaign started mode={self.config.mode} "
                f"rounds={self.config.rounds} budget={self.config.budget_per_round}"
            ),
            persist=False,
            n_remaining=len(remaining),
            initial_count=len(observed_variants),
        )

        for round_id in range(1, self.config.rounds + 1):
            self.state.round_id = round_id
            self._progress(
                "round_started",
                (
                    f"round {round_id}/{self.config.rounds} "
                    f"({len(observed_variants)} observed, {len(remaining)} remaining)"
                ),
                n_observed=len(observed_variants),
                n_remaining=len(remaining),
                rounds=self.config.rounds,
            )
            predictor = None
            if selection_driver == "predictor":
                self._progress(
                    "generation_model_fit_started",
                    f"round {round_id}/{self.config.rounds} fitting generation predictor {self.config.model.name}",
                    phase=CampaignPhase.MODEL_FIT,
                    n_train=len(observed_variants),
                    model=self.config.model.name,
                    device=self.config.model.device,
                )
                predictor = self._create_and_fit_predictor(
                    self.config.model,
                    observed_variants,
                    self.config.seed + round_id,
                )
                self._progress(
                    "generation_model_fit_completed",
                    f"round {round_id}/{self.config.rounds} generation predictor fit complete",
                    n_train=len(observed_variants),
                    model=self.config.model.name,
                )

            evidence: dict[str, list[Any]] = {}
            local_evidence: tuple[Any, ...] = ()
            if self.config.knowledge_enabled:
                self._progress(
                    "evidence_started",
                    (
                        f"round {round_id}/{self.config.rounds} scoring observed "
                        f"{len(observed_variants)} variants on all channels"
                    ),
                    n_candidates=len(observed_variants),
                )
                observed_evidence = self.knowledge.evidence_for(
                    observed_variants,
                    round_id=round_id,
                    delete_evidence=self.config.evidence_deletion,
                )
                evidence.update(observed_evidence)
                self._persist_evidence_map(observed_variants, observed_evidence)
                remaining_kg: dict[str, list[Any]] = {}
                if "kg" in self.knowledge.providers and remaining:
                    self._progress(
                        "evidence_started",
                        (
                            f"round {round_id}/{self.config.rounds} scoring remaining "
                            f"{len(remaining)} variants on kg"
                        ),
                        n_candidates=len(remaining),
                    )
                    remaining_kg = self.knowledge.evidence_for(
                        remaining,
                        round_id=round_id,
                        delete_evidence=self.config.evidence_deletion,
                        channels=("kg",),
                    )
                    for variant_id, bundle in remaining_kg.items():
                        evidence[variant_id] = bundle
                if evidence:
                    all_evidence = [item for bundle in evidence.values() for item in bundle]
                    persisted_evidence = [
                        item for bundle in observed_evidence.values() for item in bundle
                    ]
                    self.writer.write_json(
                        f"round_{round_id:02d}/evidence_contract.json",
                        {
                            "provider_status": self.knowledge.provider_status,
                            "parameter_set_id": self.config.knowledge.parameter_set_id,
                            "variant_count": len(evidence),
                            "persisted_variant_count": len(observed_evidence),
                            "remaining_kg_variant_count": len(remaining_kg),
                            "evidence_count": len(all_evidence),
                            "persisted_evidence_count": len(persisted_evidence),
                            "channel_counts": {
                                channel: sum(item.channel == channel for item in all_evidence)
                                for channel in sorted({item.channel for item in all_evidence})
                            },
                            "quality_counts": {
                                status: sum(item.quality_status == status for item in all_evidence)
                                for status in sorted({item.quality_status for item in all_evidence})
                            },
                            "calibrated_count": sum(item.calibrated for item in all_evidence),
                            "selection_eligible_count": sum(
                                item.contributes_to_selection for item in all_evidence
                            ),
                            "calibration_models": {
                                item.channel: item.provenance["calibration"]
                                for item in all_evidence
                                if item.calibrated and "calibration" in item.provenance
                            },
                        },
                    )
                    self.writer.write_json(
                        f"round_{round_id:02d}/parameter_snapshot.json",
                        {
                            "parameter_set_id": self.config.knowledge.parameter_set_id,
                            "parameters": self.config.knowledge.parameters,
                            "boundary": "round_start_after_previous_wet_reveal",
                            "round_id": round_id,
                            "visible_observation_count": len(self.state.observed),
                            "automatic_parameter_update_applied": False,
                        },
                    )
                self._progress(
                    "evidence_completed",
                    f"round {round_id}/{self.config.rounds} evidence ready",
                    n_candidates=len(observed_variants),
                )

            if self.knowledge.local_knowledge is not None:
                local_result, local_evidence = self.knowledge.prefetch_local_knowledge(
                    round_id=round_id,
                    objective=self.config.task.objective,
                    assay_conditions=self.config.task.assay_conditions,
                    anchors=tuple(sorted(self.knowledge.providers)),
                    candidates=observed_variants,
                )
                selecting_local_evidence = []
                for item in local_evidence:
                    if item.contributes_to_selection and item.variant_id in evidence:
                        evidence[item.variant_id].append(item)
                        selecting_local_evidence.append(item)
                if selecting_local_evidence:
                    self.knowledge.graph.add_evidence(selecting_local_evidence)
                self.writer.write_json(
                    f"round_{round_id:02d}/local_rag_retrieval.json",
                    local_result,
                )
                self.writer.write_json(
                    f"round_{round_id:02d}/local_rag_evidence.json",
                    local_evidence,
                )
                self.writer.event(
                    "local_knowledge_retrieved",
                    {
                        "round_id": round_id,
                        "query_id": local_result.query_id if local_result else None,
                        "chunk_count": len(local_result.chunks) if local_result else 0,
                        "evidence_count": len(local_evidence),
                        "policy_decision": (local_result.policy_decision if local_result else None),
                    },
                )

            structured_variants = _decision_ingest_variants(
                observed_variants,
                observations=self.state.observed,
            )
            structured_result = self.knowledge.sync_structured_kg(
                run_id=self.run_id,
                round_id=round_id,
                variants=structured_variants,
                observations=self.state.observed,
                evidence=[
                    *local_evidence,
                    *self._flatten_round_evidence(evidence),
                ],
                hypotheses=self.state.hypotheses,
            )
            self.writer.write_json(
                f"round_{round_id:02d}/structured_kg_pre_design.json",
                {
                    "report": structured_result.report,
                    "entity_count": len(structured_result.snapshot.entities),
                    "relation_count": len(structured_result.snapshot.relations),
                    "snapshot_id": self.knowledge.structured_sink.last_snapshot_id,
                },
            )
            self.writer.event(
                "structured_kg_built",
                {
                    "round_id": round_id,
                    "stage": "pre_design",
                    "entity_count": len(structured_result.snapshot.entities),
                    "relation_count": len(structured_result.snapshot.relations),
                },
            )
            interaction_result = self._run_kg_interaction(
                round_id=round_id,
                observed_variants=observed_variants,
            )
            if interaction_result is not None:
                self._record_kg_interaction(
                    round_id=round_id, interaction_result=interaction_result
                )

            hypothesis = None
            if self.config.mode in {"llm_agent", "knowledge_agent"}:
                self._progress(
                    "hypothesis_generation_started",
                    f"round {round_id}/{self.config.rounds} requesting scientist hypothesis",
                    phase=CampaignPhase.LLM_HYPOTHESIS,
                    model=self.config.llm.model or self.config.llm.provider,
                )
                round_evidence = self._scientist_prompt_evidence(evidence)
                scientist_evidence = [
                    *(local_evidence if self._scientist_local_context_allowed else ()),
                    *round_evidence,
                ]
                scientist_activation = self._role_activation_state(
                    "scientist",
                    evidence=round_evidence,
                    local_evidence=local_evidence,
                    interaction_result=interaction_result,
                )
                if self.hypothesis_graph is not None:
                    base_context = self.agent.sanitized_context(
                        self.state, observed_variants, self.state.observed
                    )
                    base_context["activation_state"] = scientist_activation.model_dump(
                        mode="json"
                    )

                    def propose_main_hypothesis(
                        *,
                        approved_subhypotheses,
                        cross_channel_conflicts,
                        base_interaction,
                        base_evidence,
                        critic_revision,
                        hypothesis_attempt,
                        _scientist_activation=scientist_activation,
                    ):
                        return self.agent.propose_hypothesis(
                            self.state,
                            observed_variants,
                            self.state.observed,
                            base_evidence,
                            kg_interaction=base_interaction,
                            activation_state=_scientist_activation,
                            approved_subhypotheses=[
                                item.model_dump(mode="json")
                                for item in approved_subhypotheses
                            ],
                            cross_channel_conflicts=[
                                item.model_dump(mode="json")
                                for item in cross_channel_conflicts
                            ],
                            critic_revision=critic_revision,
                            hypothesis_attempt=hypothesis_attempt,
                        )

                    pipeline_result = self.hypothesis_graph.run(
                        base_context=ScientistContextInput.model_validate(base_context),
                        evidence=scientist_evidence,
                        interaction=interaction_result,
                        main_proposer=propose_main_hypothesis,
                    )
                    self.writer.write_json(
                        f"round_{round_id:02d}/hypothesis_pipeline.json",
                        pipeline_result.model_dump(mode="json"),
                    )
                    self.writer.event(
                        "hypothesis_pipeline_completed",
                        {
                            "round_id": round_id,
                            "status": pipeline_result.status,
                            "failure_code": pipeline_result.failure_code,
                            "branch_status": {
                                item.channel: item.status for item in pipeline_result.branches
                            },
                            "main_attempts": pipeline_result.main_attempts,
                        },
                    )
                    if pipeline_result.status != "SUCCEEDED":
                        failure = pipeline_result.failure_code or "HYPOTHESIS_PIPELINE_FAILED"
                        self._required_node_failures.append(f"round_{round_id}:{failure}")
                        rounds_aborted += 1
                        self._progress(
                            None,
                            f"round {round_id} aborted by hypothesis review graph",
                            phase=CampaignPhase.ROUND_ABORTED,
                            persist=False,
                        )
                        self.writer.event(
                            "round_aborted",
                            {"round_id": round_id, "reason": failure, "decision_ids": []},
                        )
                        break
                    hypothesis = Hypothesis(**(pipeline_result.main_hypothesis or {}))
                else:
                    hypothesis = self.agent.propose_hypothesis(
                        self.state,
                        observed_variants,
                        self.state.observed,
                        scientist_evidence,
                        kg_interaction=interaction_result,
                        activation_state=scientist_activation,
                    )
                self.state.hypotheses.append(hypothesis)
                self.knowledge.graph.add_hypothesis(
                    hypothesis.hypothesis_id,
                    round_id,
                    hypothesis.statement,
                    hypothesis.evidence_ids,
                )
                self.writer.event("hypothesis_proposed", hypothesis.__dict__)
                self._progress(
                    "hypothesis_generation_completed",
                    f"round {round_id}/{self.config.rounds} hypothesis {hypothesis.hypothesis_id}",
                    hypothesis_id=hypothesis.hypothesis_id,
                )
                for query_id in self.agent.last_knowledge_query_ids:
                    self.writer.event(
                        "knowledge_graph_queried",
                        {"round_id": round_id, "query_id": query_id},
                    )

            eligible = self.generator.generate(
                remaining,
                self.state,
                hypothesis,
                evidence,
                self.config.candidate_limit,
            )
            if not eligible:
                raise RuntimeError("Candidate generator returned an empty pool")
            self._progress(
                "batch_proposed",
                f"round {round_id}/{self.config.rounds} proposed {len(eligible)} eligible candidates",
                phase=CampaignPhase.PROPOSED,
                n_candidates=len(eligible),
            )
            score_full_remaining = (
                selection_driver == "predictor" and self.config.candidate_limit <= 0
            )
            predict_targets = remaining if score_full_remaining else eligible
            expected_batch_size = min(self.config.budget_per_round, len(eligible))
            knowledge_scores = (
                self.knowledge.scores(evidence) if self.config.knowledge_enabled else {}
            )
            design_scores: list[DesignScore] = []
            generation_prediction_sets: list[list[Prediction]] = []
            active_score_result = None
            active_knowledge_scores: dict[str, float] = {}
            agent_quota_selection = None

            if selection_driver == "agent_uq":
                if self.config.generation.use_fitness_predictors:
                    for model_index, model_config in enumerate(
                        self.config.generation.predictor_models
                    ):
                        generation_predictor = self._create_and_fit_predictor(
                            model_config,
                            observed_variants,
                            self.config.seed + round_id * 101 + model_index,
                        )
                        generation_prediction_sets.append(generation_predictor.predict(eligible))
                prior_scores = self.knowledge.validation_prior_scores(
                    eligible,
                    round_id=round_id,
                )
                design_scores = self.agent_selector.score(
                    eligible,
                    observed_variants=observed_variants,
                    hypothesis=hypothesis,
                    hypotheses=self.state.hypotheses,
                    evidence=evidence,
                    prior_scores=prior_scores,
                    predictor_predictions=generation_prediction_sets,
                )
                design_predictions = self.agent_selector.as_predictions(design_scores)
                if self.config.score_shuffle:
                    design_predictions = _shuffle_prediction_scores(design_predictions, self.rng)
                all_scores = {item.variant_id: item.fitness_mean for item in design_predictions}
                working_by_id = {item.variant_id: item for item in design_predictions}
            elif selection_driver == "active_learning":
                if self.active_learning is None:
                    raise AssertionError("Active-learning selection has no configured module")
                validation_prior_scores = self.knowledge.validation_prior_scores(
                    eligible,
                    round_id=round_id,
                )
                active_knowledge_scores = {
                    item.variant_id: knowledge_scores.get(item.variant_id, 0.0)
                    + self.config.active_learning.acquisition.validation_prior_weight
                    * validation_prior_scores.get(item.variant_id, 0.0)
                    for item in eligible
                }
                self._progress(
                    "active_learning_posterior_fit_started",
                    f"round {round_id}/{self.config.rounds} fitting visible-label posterior",
                    phase=CampaignPhase.MODEL_FIT,
                    n_train=len(observed_variants),
                    n_candidates=len(eligible),
                    module=self.config.active_learning.module,
                )
                posterior_result = self.active_learning.fit_predict(
                    observed_variants,
                    self.state.observed,
                    eligible,
                )
                selection_posterior = posterior_result
                if self.config.score_shuffle:
                    selection_posterior = replace(
                        posterior_result,
                        predictions=tuple(
                            _shuffle_prediction_scores(
                                posterior_result.predictions,
                                self.rng,
                            )
                        ),
                    )
                active_score_result = self.active_learning.score(
                    selection_posterior,
                    active_knowledge_scores,
                )
                working_predictions = list(selection_posterior.predictions)
                all_scores = active_score_result.composite_by_id()
                active_scores_by_id = active_score_result.by_id()
                working_by_id = {item.variant_id: item for item in working_predictions}
                design_scores = [
                    DesignScore(
                        item.variant_id,
                        all_scores[item.variant_id],
                        item.fitness_std,
                        0.0,
                        knowledge_scores.get(item.variant_id, 0.0),
                        validation_prior_scores.get(item.variant_id, 0.0),
                        item.fitness_mean,
                        f"active_learning:{self.config.active_learning.module}",
                        (
                            f"Calibrated posterior {item.model_version}; hybrid components: "
                            f"exploitation={active_scores_by_id[item.variant_id].exploitation:.3f}; "
                            f"exploration={active_scores_by_id[item.variant_id].exploration:.3f}; "
                            f"knowledge={active_scores_by_id[item.variant_id].knowledge:.3f}; "
                            f"ood={item.ood_score:.3f}."
                        ),
                    )
                    for item in working_predictions
                ]
                self.writer.write_json(
                    f"round_{round_id:02d}/active_learning_posterior.json",
                    {
                        "module": self.config.active_learning.module,
                        "calibration": posterior_result.calibration,
                        "predictions": posterior_result.predictions,
                    },
                )
                self._progress(
                    "active_learning_posterior_ready",
                    f"round {round_id}/{self.config.rounds} calibrated posterior ready",
                    phase=CampaignPhase.PREDICTING,
                    n_candidates=len(working_predictions),
                    calibration_status=posterior_result.calibration.status,
                    calibration_observations=(
                        posterior_result.calibration.calibration_observations
                    ),
                )
            elif selection_driver == "random":
                design_scores = [
                    DesignScore(
                        item.variant_id,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        "random",
                        "Random baseline selection; no fitness or KG utility used.",
                    )
                    for item in eligible
                ]
                design_predictions = self.agent_selector.as_predictions(design_scores)
                all_scores = self.policy.score(design_predictions, {}, self.rng)
                working_by_id = {item.variant_id: item for item in design_predictions}
            else:
                if predictor is None:
                    raise AssertionError("Predictor selection driver has no fitted predictor")
                self._progress(
                    "generation_predict_started",
                    f"round {round_id}/{self.config.rounds} predictor scoring {len(predict_targets)} candidates",
                    phase=CampaignPhase.PREDICTING,
                    n_candidates=len(predict_targets),
                    model=self.config.model.name,
                )
                generation_predictions = predictor.predict(predict_targets)
                working_predictions = (
                    _shuffle_prediction_scores(generation_predictions, self.rng)
                    if self.config.score_shuffle
                    else generation_predictions
                )
                all_scores = self.policy.score(working_predictions, knowledge_scores, self.rng)
                working_by_id = {item.variant_id: item for item in working_predictions}
                design_scores = [
                    DesignScore(
                        item.variant_id,
                        all_scores[item.variant_id],
                        item.fitness_std,
                        0.0,
                        knowledge_scores.get(item.variant_id, 0.0),
                        0.0,
                        item.fitness_mean,
                        "predictor",
                        f"Fitness-direct baseline using {item.model_version} and {self.policy.name} acquisition.",
                    )
                    for item in working_predictions
                ]

            design_score_by_id = {item.variant_id: item for item in design_scores}
            self.writer.write_json(f"round_{round_id:02d}/design_scores.json", design_scores)
            self._progress(
                "design_scored",
                f"round {round_id}/{self.config.rounds} design utilities ready via {selection_driver}",
                phase=CampaignPhase.DESIGN_SCORED,
                n_candidates=len(eligible),
            )
            if selection_driver == "active_learning":
                if self.active_learning is None or active_score_result is None:
                    raise AssertionError("Active-learning acquisition is unavailable")
                active_selection = self.active_learning.select(
                    eligible,
                    active_score_result,
                    expected_batch_size,
                    knowledge_scores=active_knowledge_scores,
                )
                selected_ids = list(active_selection.selected_ids)
                self.writer.write_json(
                    f"round_{round_id:02d}/active_learning_acquisition.json",
                    {
                        "module": self.config.active_learning.module,
                        "candidate_scores": active_score_result.scores,
                        "selection": active_selection,
                    },
                )
                self.writer.event(
                    "active_learning_batch_selected",
                    {
                        "round_id": round_id,
                        "module": self.config.active_learning.module,
                        "selected_ids": selected_ids,
                        "quotas": active_selection.quotas,
                        "selected_by_arm": active_selection.selected_by_arm,
                    },
                )
            elif selection_driver == "agent_uq" and self.agent_quota_acquisition is not None:
                agent_quota_selection = self.agent_quota_acquisition.select(
                    eligible,
                    [
                        replace(
                            item,
                            utility=all_scores[item.variant_id],
                        )
                        for item in design_scores
                    ],
                    expected_batch_size,
                    diversity_lambda=self.config.diversity_lambda,
                )
                selected_ids = list(agent_quota_selection.selected_ids)
                self.writer.write_json(
                    f"round_{round_id:02d}/agent_quota_acquisition.json",
                    agent_quota_selection,
                )
                self.writer.event(
                    "agent_quota_batch_selected",
                    {
                        "round_id": round_id,
                        "plugin": agent_quota_selection.plugin,
                        "selected_ids": selected_ids,
                        "quotas": agent_quota_selection.quotas,
                        "selected_by_arm": agent_quota_selection.selected_by_arm,
                        "matched_control_pairs": (agent_quota_selection.matched_control_pairs),
                        "shortfalls": agent_quota_selection.shortfalls,
                        "fallback_ids": agent_quota_selection.fallback_ids,
                    },
                )
            else:
                selected_ids = self.policy.select(
                    eligible,
                    [working_by_id[item.variant_id] for item in eligible],
                    all_scores,
                    expected_batch_size,
                    self.config.diversity_lambda,
                )
            if len(selected_ids) != min(self.config.budget_per_round, len(eligible)):
                raise RuntimeError("Acquisition returned an incomplete batch")
            self.writer.event(
                "batch_initial_selected",
                {
                    "round_id": round_id,
                    "selection_driver": selection_driver,
                    "candidate_ids": selected_ids,
                    "fitness_predictors_used_for_generation": (
                        self._fitness_predictors_used_for_generation(selection_driver)
                    ),
                },
            )
            if evidence:
                self.writer.write_json(
                    f"round_{round_id:02d}/selected_evidence.json",
                    {variant_id: evidence.get(variant_id, []) for variant_id in selected_ids},
                )

            validation_prediction_sets: list[list[Prediction]] = []
            validation_predictors: list[Any] = []
            validation_configs = (
                (self.config.model, *self.config.validation.predictor_models)
                if self.config.validation.enabled
                else ()
            )
            if validation_configs:
                self._progress(
                    "model_fit_started",
                    f"round {round_id}/{self.config.rounds} preparing dry validation models",
                    phase=CampaignPhase.MODEL_FIT,
                    n_train=len(observed_variants),
                    model_count=len(validation_configs),
                )
            for model_index, model_config in enumerate(validation_configs):
                if model_index == 0 and predictor is not None:
                    validation_predictor = predictor
                else:
                    self._progress(
                        "validation_model_fit_started",
                        f"round {round_id}/{self.config.rounds} fitting dry validator {model_config.name}",
                        phase=CampaignPhase.MODEL_FIT,
                        n_train=len(observed_variants),
                        model=model_config.name,
                    )
                    validation_predictor = self._create_and_fit_predictor(
                        model_config,
                        observed_variants,
                        self.config.seed + round_id * 1009 + model_index,
                    )
                    self._progress(
                        "validation_model_fit_completed",
                        f"round {round_id}/{self.config.rounds} dry validator fit complete",
                        model=model_config.name,
                    )
                validation_predictors.append(validation_predictor)
            if validation_predictors:
                self._progress(
                    "model_fit_completed",
                    f"round {round_id}/{self.config.rounds} dry validation models ready",
                    n_train=len(observed_variants),
                    model_count=len(validation_predictors),
                )
                self._progress(
                    "predict_started",
                    f"round {round_id}/{self.config.rounds} validating {len(predict_targets)} candidates",
                    phase=CampaignPhase.PREDICTING,
                    n_candidates=len(predict_targets),
                    model_count=len(validation_predictors),
                )
                validation_prediction_sets = [
                    validation_predictor.predict(predict_targets)
                    for validation_predictor in validation_predictors
                ]
                self._progress(
                    "predict_completed",
                    f"round {round_id}/{self.config.rounds} dry validation predictions ready",
                    n_candidates=len(predict_targets),
                    model_count=len(validation_predictors),
                )
            if validation_prediction_sets:
                original_predictions = validation_prediction_sets[0]
            else:
                original_predictions = [working_by_id[item.variant_id] for item in predict_targets]
            prediction_by_id = {item.variant_id: item for item in original_predictions}
            self._progress(
                "dry_validation_completed",
                f"round {round_id}/{self.config.rounds} dry validation ready",
                phase=CampaignPhase.DRY_VALIDATED,
                n_candidates=len(original_predictions),
                model_versions=",".join(
                    sorted({item.model_version for item in original_predictions})
                ),
            )

            inference_interventions = tuple(
                tag
                for enabled, tag in (
                    (self.config.score_shuffle, "score_shuffle"),
                    (self.config.evidence_deletion, "evidence_deletion"),
                )
                if enabled
            )
            if self.config.knowledge_enabled:
                self.knowledge.record_inference_context(
                    predict_targets,
                    original_predictions,
                    {
                        item.variant_id: evidence.get(item.variant_id, [])
                        for item in predict_targets
                    },
                    round_id=round_id,
                    intervention_tags=inference_interventions,
                )

            initial_selected_ids = tuple(selected_ids)
            draft_context = {
                "initial_selected_ids": initial_selected_ids,
                "eligible": eligible,
                "working_by_id": working_by_id,
                "all_scores": all_scores,
                "expected_batch_size": expected_batch_size,
                "active_score_result": active_score_result,
                "active_knowledge_scores": active_knowledge_scores,
                "agent_quota_selection": agent_quota_selection,
                "predictor": validation_predictors[0] if validation_predictors else None,
                "prediction_by_id": prediction_by_id,
                "round_id": round_id,
                "evidence": evidence,
                "hypothesis": hypothesis,
                "rationale_claims": {item.variant_id: item.reason for item in design_scores},
            }

            def draft_builder(
                review_attempt: int,
                parent_draft_batch_id: str | None,
                exclusions: set[str],
                constraints: RevisionConstraints | None = None,
                _context: dict[str, Any] = draft_context,
                _design_score_by_id: dict[str, DesignScore] = design_score_by_id,
            ):
                _context["revision_constraints"] = constraints
                if review_attempt == 0:
                    candidate_ids = list(_context["initial_selected_ids"])
                else:
                    revised_eligible = [
                        item for item in _context["eligible"] if item.variant_id not in exclusions
                    ]
                    if selection_driver == "active_learning":
                        if self.active_learning is None or _context["active_score_result"] is None:
                            raise AssertionError(
                                "Active-learning acquisition is unavailable during revision"
                            )
                        revised_selection = self.active_learning.select(
                            revised_eligible,
                            _context["active_score_result"],
                            min(_context["expected_batch_size"], len(revised_eligible)),
                            knowledge_scores=_context["active_knowledge_scores"],
                        )
                        candidate_ids = list(revised_selection.selected_ids)
                    elif (
                        selection_driver == "agent_uq" and self.agent_quota_acquisition is not None
                    ):
                        diversity = self.config.diversity_lambda
                        if constraints is not None and constraints.increase_diversity:
                            diversity = max(diversity, diversity + 0.15, 0.25)
                        revised_selection = self.agent_quota_acquisition.select(
                            revised_eligible,
                            [
                                replace(
                                    _design_score_by_id[item.variant_id],
                                    utility=_context["all_scores"][item.variant_id],
                                )
                                for item in revised_eligible
                            ],
                            min(_context["expected_batch_size"], len(revised_eligible)),
                            diversity_lambda=diversity,
                            constraints=constraints,
                        )
                        _context["agent_quota_selection"] = revised_selection
                        candidate_ids = list(revised_selection.selected_ids)
                    else:
                        diversity = self.config.diversity_lambda
                        if constraints is not None and constraints.increase_diversity:
                            diversity = max(diversity, diversity + 0.15, 0.25)
                        candidate_ids = self.policy.select(
                            revised_eligible,
                            [
                                _context["working_by_id"][item.variant_id]
                                for item in revised_eligible
                            ],
                            _context["all_scores"],
                            min(_context["expected_batch_size"], len(revised_eligible)),
                            diversity,
                        )
                candidate_variants = [public_by_id[item] for item in candidate_ids]
                missing_predictions = [
                    item
                    for item in candidate_variants
                    if item.variant_id not in _context["prediction_by_id"]
                ]
                if missing_predictions and _context["predictor"] is not None:
                    refreshed_predictions = _context["predictor"].predict(missing_predictions)
                    _context["prediction_by_id"].update(
                        {item.variant_id: item for item in refreshed_predictions}
                    )
                if self.config.knowledge_enabled:
                    refreshed_evidence = self.knowledge.evidence_for(
                        candidate_variants,
                        round_id=_context["round_id"],
                        delete_evidence=self.config.evidence_deletion,
                    )
                    _context["evidence"].update(refreshed_evidence)
                    self._persist_evidence_map(candidate_variants, refreshed_evidence)
                falsification_spec = (
                    preregister_batch_median_test(
                        hypothesis_id=_context["hypothesis"].hypothesis_id,
                        round_id=_context["round_id"],
                        target_variant_ids=candidate_ids,
                        visible_observations=self.state.observed,
                    )
                    if _context["hypothesis"] is not None
                    else None
                )
                return build_draft_batch(
                    round_id=_context["round_id"],
                    review_attempt=review_attempt,
                    candidate_ids=candidate_ids,
                    variants=public_by_id,
                    predictions=_context["prediction_by_id"],
                    evidence=_context["evidence"],
                    hypothesis_id=(
                        _context["hypothesis"].hypothesis_id if _context["hypothesis"] else None
                    ),
                    falsification_spec=falsification_spec,
                    parent_draft_batch_id=parent_draft_batch_id,
                    rationale_claims=_context["rationale_claims"],
                )

            def record_review_start(draft, report, _round_id: int = round_id) -> None:
                folder = f"round_{_round_id:02d}"
                self._progress(
                    "review_attempt_started",
                    (
                        f"round {_round_id} review attempt {draft.review_attempt} "
                        f"(hard_conflicts={len(report.hard_conflicts)})"
                    ),
                    phase=CampaignPhase.HARD_VALIDATED,
                    attempt=draft.review_attempt,
                    draft_batch_id=draft.draft_batch_id,
                    batch_hash=draft.batch_hash,
                    hard_conflicts=len(report.hard_conflicts),
                    conflict_count=len(report.conflicts),
                    max_attempts=self.config.critic.max_revision_attempts,
                )
                self.writer.event(
                    "batch_drafted",
                    {
                        "round_id": _round_id,
                        "attempt": draft.review_attempt,
                        "draft_batch_id": draft.draft_batch_id,
                        "batch_hash": draft.batch_hash,
                    },
                )
                self.writer.event(
                    "hard_validation_completed",
                    {
                        "round_id": _round_id,
                        "attempt": draft.review_attempt,
                        "report_id": report.report_id,
                        "hard_conflicts": len(report.hard_conflicts),
                        "conflict_count": len(report.conflicts),
                    },
                )
                if draft.falsification_spec is not None:
                    self.writer.event(
                        "falsification_registered",
                        {
                            "round_id": _round_id,
                            "attempt": draft.review_attempt,
                            "spec_id": draft.falsification_spec.spec_id,
                            "pre_registration_hash": (
                                draft.falsification_spec.pre_registration_hash
                            ),
                        },
                    )
                self.writer.write_json(
                    f"{folder}/draft_batch_attempt_{draft.review_attempt}.json", draft
                )
                self.writer.write_json(
                    f"{folder}/hard_validation_attempt_{draft.review_attempt}.json", report
                )
                self._progress(
                    "critique_started",
                    f"round {_round_id} critic review attempt {draft.review_attempt}",
                    phase=CampaignPhase.CRITIQUE_REQUESTED,
                    attempt=draft.review_attempt,
                    critic_provider=self.critic_agent.client.provider_name,
                )

            def record_review_attempt(draft, report, decision, _round_id: int = round_id) -> None:
                self.state.critique_decisions.append(decision)
                folder = f"round_{_round_id:02d}"
                self.writer.write_json(
                    f"{folder}/critique_attempt_{draft.review_attempt}.json", decision
                )
                self.writer.event(
                    "critique_completed",
                    {
                        "round_id": _round_id,
                        "attempt": draft.review_attempt,
                        "draft_batch_id": draft.draft_batch_id,
                        "batch_hash": draft.batch_hash,
                        "hard_conflicts": len(report.hard_conflicts),
                        "decision": decision,
                        "critic_provider": self.critic_agent.client.provider_name,
                        "critic_profile": self.config.critic.profile,
                    },
                )
                self._progress(
                    None,
                    (
                        f"round {_round_id} critique attempt {draft.review_attempt} "
                        f"{decision.verdict.value}"
                    ),
                    persist=False,
                    attempt=draft.review_attempt,
                )
                if decision.verdict.value == "REVISE":
                    self._progress(
                        None,
                        f"round {_round_id} critic requested revision",
                        phase=CampaignPhase.REVISION_REQUESTED,
                        persist=False,
                        attempt=draft.review_attempt,
                    )
                    self.writer.event("batch_revision_requested", decision.__dict__)

            try:
                review_result = self.review_loop.run(
                    draft_builder=draft_builder,
                    variants=public_by_id,
                    predictions=prediction_by_id,
                    evidence=evidence,
                    revealed_ids=set(self.state.revealed_variant_ids),
                    pending_ids=set(),
                    allowed_ids={item.variant_id for item in remaining},
                    expected_batch_size=expected_batch_size,
                    context_evidence=(local_evidence if self._critic_local_context_allowed else ()),
                    hypothesis=hypothesis,
                    activation_state=self._role_activation_state(
                        "critic",
                        evidence=self._flatten_round_evidence(evidence),
                        local_evidence=local_evidence,
                        interaction_result=interaction_result,
                    ),
                    on_attempt=record_review_attempt,
                    on_attempt_start=record_review_start,
                )
            except HypothesisRevisionRequested as requested:
                try:
                    if hypothesis is None:
                        raise ReviewRejected(
                            "Critic revision limit exhausted",
                            decisions=requested.decisions,
                        ) from requested
                    hypothesis = self._repropose_after_critic(
                        requested.decision,
                        hypothesis,
                        observed_variants=observed_variants,
                        evidence=evidence,
                        local_evidence=local_evidence,
                        interaction_result=interaction_result,
                    )
                    eligible = self.generator.generate(
                        remaining,
                        self.state,
                        hypothesis,
                        evidence,
                        self.config.candidate_limit,
                    )
                    if not eligible:
                        raise RuntimeError("Candidate generator returned an empty pool")
                    expected_batch_size = min(self.config.budget_per_round, len(eligible))
                    if selection_driver == "agent_uq":
                        prior_scores = self.knowledge.validation_prior_scores(
                            eligible, round_id=round_id
                        )
                        design_scores = self.agent_selector.score(
                            eligible,
                            observed_variants=observed_variants,
                            hypothesis=hypothesis,
                            hypotheses=self.state.hypotheses,
                            evidence=evidence,
                            prior_scores=prior_scores,
                            predictor_predictions=[],
                        )
                        design_predictions = self.agent_selector.as_predictions(design_scores)
                        all_scores = {
                            item.variant_id: item.fitness_mean for item in design_predictions
                        }
                        working_by_id = {item.variant_id: item for item in design_predictions}
                        design_score_by_id = {item.variant_id: item for item in design_scores}
                        if self.agent_quota_acquisition is not None:
                            agent_quota_selection = self.agent_quota_acquisition.select(
                                eligible,
                                [
                                    replace(item, utility=all_scores[item.variant_id])
                                    for item in design_scores
                                ],
                                expected_batch_size,
                                diversity_lambda=self.config.diversity_lambda,
                            )
                            selected_ids = list(agent_quota_selection.selected_ids)
                        else:
                            selected_ids = self.policy.select(
                                eligible,
                                [working_by_id[item.variant_id] for item in eligible],
                                all_scores,
                                expected_batch_size,
                                self.config.diversity_lambda,
                            )
                    else:
                        selected_ids = self.policy.select(
                            eligible,
                            [
                                working_by_id[item.variant_id]
                                for item in eligible
                                if item.variant_id in working_by_id
                            ],
                            all_scores,
                            expected_batch_size,
                            self.config.diversity_lambda,
                        )
                    draft_context.update(
                        {
                            "hypothesis": hypothesis,
                            "eligible": eligible,
                            "working_by_id": working_by_id,
                            "all_scores": all_scores,
                            "expected_batch_size": expected_batch_size,
                            "agent_quota_selection": agent_quota_selection,
                            "initial_selected_ids": tuple(selected_ids),
                            "rationale_claims": {
                                item.variant_id: item.reason for item in design_scores
                            },
                        }
                    )
                    self.writer.event(
                        "hypothesis_regenerated",
                        {
                            "round_id": round_id,
                            "hypothesis_id": hypothesis.hypothesis_id,
                            "parent_hypothesis_id": hypothesis.parent_hypothesis_id,
                            "critic_decision_id": requested.decision.decision_id,
                        },
                    )
                    review_result = self.review_loop.run(
                        draft_builder=draft_builder,
                        variants=public_by_id,
                        predictions=prediction_by_id,
                        evidence=evidence,
                        revealed_ids=set(self.state.revealed_variant_ids),
                        pending_ids=set(),
                        allowed_ids={item.variant_id for item in remaining},
                        expected_batch_size=expected_batch_size,
                        context_evidence=(
                            local_evidence if self._critic_local_context_allowed else ()
                        ),
                        hypothesis=hypothesis,
                        activation_state=self._role_activation_state(
                            "critic",
                            evidence=self._flatten_round_evidence(evidence),
                            local_evidence=local_evidence,
                            interaction_result=interaction_result,
                        ),
                        on_attempt=record_review_attempt,
                        on_attempt_start=record_review_start,
                    )
                except (HypothesisRevisionRequested, ReviewRejected) as nested:
                    error = (
                        nested
                        if isinstance(nested, ReviewRejected)
                        and not isinstance(nested, HypothesisRevisionRequested)
                        else ReviewRejected(
                            "Critic revision limit exhausted",
                            decisions=getattr(nested, "decisions", requested.decisions),
                        )
                    )
                    raise error from nested
            except ReviewRejected as error:
                terminal_policy = (
                    self.config.critic.on_exhausted
                    if "limit exhausted" in str(error)
                    else self.config.critic.on_reject
                )
                if terminal_policy == "safe_fallback":
                    safest = sorted(
                        eligible,
                        key=lambda item: (
                            working_by_id[item.variant_id].ood_score,
                            -working_by_id[item.variant_id].fitness_mean,
                            item.variant_id,
                        ),
                    )[:expected_batch_size]
                    fallback_ids = tuple(item.variant_id for item in safest)
                    fallback_context = {
                        "hypothesis": hypothesis,
                        "round_id": round_id,
                        "prediction_by_id": prediction_by_id,
                        "evidence": evidence,
                    }

                    def fallback_builder(
                        review_attempt: int,
                        parent_draft_batch_id: str | None,
                        exclusions: set[str],
                        _fallback_ids: tuple[str, ...] = fallback_ids,
                        _context: dict[str, Any] = fallback_context,
                        _design_score_by_id: dict[str, DesignScore] = design_score_by_id,
                    ):
                        candidate_ids = tuple(
                            item for item in _fallback_ids if item not in exclusions
                        )
                        fallback_spec = (
                            preregister_batch_median_test(
                                hypothesis_id=_context["hypothesis"].hypothesis_id,
                                round_id=_context["round_id"],
                                target_variant_ids=candidate_ids,
                                visible_observations=self.state.observed,
                            )
                            if _context["hypothesis"] is not None
                            else None
                        )
                        return build_draft_batch(
                            round_id=_context["round_id"],
                            review_attempt=review_attempt,
                            candidate_ids=candidate_ids,
                            variants=public_by_id,
                            predictions=_context["prediction_by_id"],
                            evidence=_context["evidence"],
                            hypothesis_id=(
                                _context["hypothesis"].hypothesis_id
                                if _context["hypothesis"]
                                else None
                            ),
                            falsification_spec=fallback_spec,
                            parent_draft_batch_id=parent_draft_batch_id,
                            rationale_claims={
                                item: _design_score_by_id[item].reason
                                for item in candidate_ids
                                if item in _design_score_by_id
                            },
                        )

                    fallback_loop = BoundedReviewLoop(
                        validator=self.hard_validator,
                        critic=CriticAgent(RuleBasedCriticClient(), max_retries=0),
                        max_revision_attempts=0,
                        gateway=self.approval_gateway,
                    )
                    try:
                        review_result = fallback_loop.run(
                            draft_builder=fallback_builder,
                            variants=public_by_id,
                            predictions=prediction_by_id,
                            evidence=evidence,
                            revealed_ids=set(self.state.revealed_variant_ids),
                            pending_ids=set(),
                            allowed_ids={item.variant_id for item in remaining},
                            expected_batch_size=expected_batch_size,
                            hypothesis=hypothesis,
                            on_attempt=record_review_attempt,
                            on_attempt_start=record_review_start,
                        )
                        self.writer.event(
                            "critic_fallback_used",
                            {
                                "round_id": round_id,
                                "policy": "lowest_ood_then_fitness",
                                "trigger": str(error),
                                "candidate_ids": fallback_ids,
                            },
                        )
                    except ReviewRejected as fallback_error:
                        error = fallback_error
                        terminal_policy = "abort_round"
                if terminal_policy == "abort_round":
                    rounds_aborted += 1
                    self._progress(
                        None,
                        f"round {round_id} aborted",
                        phase=CampaignPhase.ROUND_ABORTED,
                        persist=False,
                    )
                    self.writer.event(
                        "round_aborted",
                        {
                            "round_id": round_id,
                            "reason": str(error),
                            "decision_ids": [item.decision_id for item in error.decisions],
                        },
                    )
                    break

            approved_batch = review_result.approved_batch
            selected_ids = list(approved_batch.candidate_ids)
            final_agent_quota_selection = draft_context.get("agent_quota_selection")
            if final_agent_quota_selection is not None:
                self.writer.write_json(
                    f"round_{round_id:02d}/agent_quota_acquisition_approved.json",
                    {
                        "selection": final_agent_quota_selection,
                        "approved_candidate_ids": selected_ids,
                        "matches_approved_batch": (
                            tuple(selected_ids) == tuple(final_agent_quota_selection.selected_ids)
                        ),
                    },
                )
            self.state.approved_batch_ids.append(approved_batch.draft_batch_id)
            self._progress(
                None,
                f"round {round_id} batch approved ({len(selected_ids)} variants)",
                phase=CampaignPhase.APPROVED,
                persist=False,
            )
            self.writer.write_json(f"round_{round_id:02d}/approved_batch.json", approved_batch)
            self.writer.event("batch_approved", approved_batch.__dict__)

            model_ranks = _descending_ranks(
                {item.variant_id: item.fitness_mean for item in original_predictions}
            )
            acquisition_ranks = _descending_ranks(all_scores)
            eligible_ranks = _descending_ranks(
                {item.variant_id: all_scores[item.variant_id] for item in eligible}
            )
            intervention_tags = tuple(
                tag
                for enabled, tag in (
                    (self.config.score_shuffle, "score_shuffle"),
                    (self.config.evidence_deletion, "evidence_deletion"),
                    (not self.config.knowledge_enabled, "knowledge_ablation"),
                )
                if enabled
            )
            records: list[SelectionRecord] = []
            for order, variant_id in enumerate(selected_ids, start=1):
                prediction = prediction_by_id[variant_id]
                bundle = evidence.get(variant_id, [])
                records.append(
                    SelectionRecord(
                        variant_id=variant_id,
                        round_id=round_id,
                        selection_order=order,
                        model_rank_all=model_ranks[variant_id],
                        acquisition_rank_all=acquisition_ranks[variant_id],
                        eligible_rank=eligible_ranks[variant_id],
                        total_candidates=len(remaining),
                        eligible_candidates=len(eligible),
                        fitness_mean=prediction.fitness_mean,
                        fitness_std=prediction.fitness_std,
                        acquisition_score=all_scores[variant_id],
                        knowledge_score=knowledge_scores.get(variant_id, 0.0),
                        evidence_ids=tuple(item.evidence_id for item in bundle),
                        hypothesis_id=hypothesis.hypothesis_id if hypothesis else None,
                        reason=(
                            f"{design_score_by_id[variant_id].reason} "
                            f"Critic: {review_result.decision.summary}"
                        ),
                        intervention_tags=intervention_tags,
                        selection_driver=design_score_by_id[variant_id].selection_driver,
                        design_score=design_score_by_id[variant_id].utility,
                        design_uncertainty=design_score_by_id[variant_id].uncertainty,
                        validation_model_versions=tuple(
                            predictions[0].model_version
                            for predictions in validation_prediction_sets
                            if predictions
                        ),
                    )
                )
            self.state.selections.extend(records)
            self._progress(
                None,
                f"round {round_id} selected {len(records)} variants",
                phase=CampaignPhase.SELECTED,
                persist=False,
            )
            self.writer.write_selection(round_id, records)
            self.writer.event(
                "batch_selected",
                {
                    "round_id": round_id,
                    "records": records,
                    "global_rank_definition": (
                        "model_rank_all ranks predictor mean over scored candidates "
                        "(the candidate pool when candidate_limit > 0, otherwise the full "
                        "unobserved oracle pool); acquisition_rank_all ranks the active policy "
                        "over that same scored set"
                    ),
                },
            )

            final_report = self.hard_validator.validate(
                review_result.draft,
                variants=public_by_id,
                predictions=prediction_by_id,
                evidence=evidence,
                revealed_ids=set(self.state.revealed_variant_ids),
                pending_ids=set(),
                allowed_ids={item.variant_id for item in remaining},
                expected_batch_size=expected_batch_size,
            )
            if final_report.hard_conflicts or final_report.input_hash != approved_batch.batch_hash:
                raise PermissionError("Final validation no longer matches the approved batch")
            experiment_run_id = self.backend.submit(approved_batch)
            self._progress(
                None,
                f"round {round_id} submitted to oracle",
                phase=CampaignPhase.SUBMITTED,
                persist=False,
            )
            revealed = self.backend.collect(experiment_run_id)
            selected_variants = [public_by_id[variant_id] for variant_id in selected_ids]
            pre_round_visible_baseline = float(
                np.median([item.fitness for item in self.state.observed])
            )
            self.state.observed.extend(revealed)
            observed_variants.extend(selected_variants)
            self.state.revealed_variant_ids.update(selected_ids)
            self.knowledge.update(selected_variants, revealed)
            revealed_by_id = {item.variant_id: item for item in revealed}
            selection_by_id = {item.variant_id: item for item in records}
            dry_by_variant: dict[str, list[Prediction]] = {
                variant_id: [] for variant_id in selected_ids
            }
            for predictions in validation_prediction_sets:
                mapping = {item.variant_id: item for item in predictions}
                for variant_id in selected_ids:
                    if variant_id in mapping:
                        dry_by_variant[variant_id].append(mapping[variant_id])
            rethink_context = ReThinkContextInput.model_validate(
                {
                    "run_id": self.run_id,
                    "round_id": round_id,
                    "visible_baseline": pre_round_visible_baseline,
                    "activation_state": self._role_activation_state(
                        "rethink",
                        interaction_result=interaction_result,
                    ).model_dump(mode="json"),
                    "approved_hypothesis": (
                        {
                            "hypothesis_id": hypothesis.hypothesis_id,
                            "statement": hypothesis.statement,
                            "expected_outcome": hypothesis.expected_outcome,
                            "falsification_criterion": hypothesis.falsification_criterion,
                            "evidence_ids": list(hypothesis.evidence_ids),
                            "explanation": hypothesis.explanation,
                        }
                        if hypothesis is not None
                        else None
                    ),
                    "final_critic_decision": {
                        "decision_id": review_result.decision.decision_id,
                        "verdict": review_result.decision.verdict.value,
                        "summary": review_result.decision.summary,
                        "cited_evidence_ids": list(
                            review_result.decision.cited_evidence_ids
                        ),
                    },
                    "candidates": [
                        {
                            "variant_id": variant_id,
                            "mutation_notation": public_by_id[variant_id].mutation_notation,
                            "agent_reason": selection_by_id[variant_id].reason,
                            "evidence_ids": list(selection_by_id[variant_id].evidence_ids),
                            "wet_value": revealed_by_id[variant_id].fitness,
                            "dry_validations": [
                                {
                                    "value": prediction.fitness_mean,
                                    "uncertainty": prediction.fitness_std,
                                    "ood_score": prediction.ood_score,
                                    "model_version": prediction.model_version,
                                }
                                for prediction in dry_by_variant[variant_id]
                            ],
                        }
                        for variant_id in selected_ids
                    ],
                }
            )
            reflections = ()
            if self.config.validation.rethink_enabled:
                try:
                    reflections = self.rethink_client.reflect_round(context=rethink_context)
                    if {item.variant_id for item in reflections} != set(selected_ids):
                        raise ValueError("ReThink output did not cover every selected variant")
                except Exception as error:  # noqa: BLE001 - provider boundary must degrade safely
                    self._fallback_nodes.append(f"round_{round_id}:rethink")
                    self.writer.event(
                        "rethink_fallback_used",
                        {"round_id": round_id, "error": str(error)},
                    )
                    reflections = create_rethink_client("mock").reflect_round(
                        context=rethink_context
                    )
            reflection_by_id = {item.variant_id: item for item in reflections}
            self.state.rethink_reflections.extend(reflections)
            current_validation_records: list[ValidationRecord] = []
            for variant_id in selected_ids:
                variant = public_by_id[variant_id]
                selection = selection_by_id[variant_id]
                reflection = reflection_by_id.get(variant_id)
                wet_observation = revealed_by_id[variant_id]
                wet_source = f"wet:{wet_observation.source}"
                current_validation_records.append(
                    ValidationRecord(
                        record_id=_validation_record_id(
                            self.run_id, round_id, variant_id, "wet", wet_source
                        ),
                        variant_id=variant_id,
                        round_id=round_id,
                        validation_type="wet",
                        mutation_notation=variant.mutation_notation,
                        value=wet_observation.fitness,
                        uncertainty=0.0,
                        source_id=wet_source,
                        model_version=None,
                        base_weight=self.config.validation.wet_weight,
                        reliability=1.0,
                        agent_reason=selection.reason,
                        hypothesis_id=selection.hypothesis_id,
                        evidence_ids=selection.evidence_ids,
                        reflection_id=(reflection.reflection_id if reflection else None),
                        reflection_verdict=(reflection.verdict if reflection else None),
                        reflection_summary=(reflection.summary if reflection else ""),
                    )
                )
                for prediction in dry_by_variant[variant_id]:
                    historical_reliability = self.knowledge.graph.dry_model_reliability(
                        prediction.model_version,
                        floor=self.config.validation.dry_reliability_floor,
                    )
                    reliability = max(
                        self.config.validation.dry_reliability_floor,
                        historical_reliability * max(0.0, 1.0 - prediction.ood_score),
                    )
                    dry_source = f"dry:{prediction.model_version}"
                    current_validation_records.append(
                        ValidationRecord(
                            record_id=_validation_record_id(
                                self.run_id,
                                round_id,
                                variant_id,
                                "dry",
                                dry_source,
                            ),
                            variant_id=variant_id,
                            round_id=round_id,
                            validation_type="dry",
                            mutation_notation=variant.mutation_notation,
                            value=prediction.fitness_mean,
                            uncertainty=prediction.fitness_std,
                            source_id=dry_source,
                            model_version=prediction.model_version,
                            base_weight=self.config.validation.dry_weight_cap,
                            reliability=reliability,
                            agent_reason=selection.reason,
                            hypothesis_id=selection.hypothesis_id,
                            evidence_ids=selection.evidence_ids,
                            reflection_id=(reflection.reflection_id if reflection else None),
                            reflection_verdict=(reflection.verdict if reflection else None),
                            reflection_summary=(reflection.summary if reflection else ""),
                        )
                    )
            self.validation_records.extend(current_validation_records)
            self.knowledge.record_validation(
                current_validation_records,
                reflections,
            )
            self.writer.write_json(
                f"round_{round_id:02d}/validation_matrix.json",
                current_validation_records,
            )
            self.writer.write_csv(
                f"round_{round_id:02d}/validation_matrix.csv",
                [item.__dict__ for item in current_validation_records],
            )
            self.writer.write_json(
                f"round_{round_id:02d}/rethink.json",
                reflections,
            )
            self._progress(
                "rethink_completed",
                f"round {round_id}/{self.config.rounds} ReThink and validation KG update complete",
                phase=CampaignPhase.RETHOUGHT,
                reflection_count=len(reflections),
                validation_record_count=len(current_validation_records),
            )
            selected_set = set(selected_ids)
            remaining = [item for item in remaining if item.variant_id not in selected_set]
            metrics = loop_round_metrics(
                self.state.observed,
                revealed,
                total_pool_size=len(original_predictions),
                selected_model_ranks=[record.model_rank_all for record in records],
            )
            metrics["round_id"] = float(round_id)
            round_metrics.append(metrics)
            self._progress(
                "round_completed",
                (
                    f"round {round_id}/{self.config.rounds} complete "
                    f"batch_best={metrics['batch_best_fitness']:.3f}"
                ),
                phase=CampaignPhase.MEASURED,
                batch_best_fitness=float(metrics["batch_best_fitness"]),
            )
            self.writer.write_json(f"round_{round_id:02d}/metrics.json", metrics)
            self.writer.event(
                "batch_measured",
                {
                    "round_id": round_id,
                    "observations": revealed,
                    "metrics": metrics,
                },
            )
            if hypothesis is not None and review_result.draft.falsification_spec is not None:
                assessment = self.hypothesis_evaluator.evaluate(
                    spec=review_result.draft.falsification_spec,
                    observations=self.state.observed,
                    round_id=round_id,
                )
                self.state.hypothesis_assessments.append(assessment)
                self._progress(
                    None,
                    f"round {round_id} hypothesis {assessment.status.value}",
                    phase=CampaignPhase.HYPOTHESIS_EVALUATED,
                    persist=False,
                )
                self.writer.write_json(
                    f"round_{round_id:02d}/hypothesis_assessment.json", assessment
                )
                self.writer.event("hypothesis_assessed", assessment.__dict__)

            post_structured_result = self.knowledge.sync_structured_kg(
                run_id=self.run_id,
                round_id=round_id,
                variants=_decision_ingest_variants(
                    observed_variants,
                    selected_variants=selected_variants,
                    observations=self.state.observed,
                ),
                observations=self.state.observed,
                predictions=[
                    prediction
                    for predictions in validation_prediction_sets
                    for prediction in predictions
                    if prediction.variant_id in selected_set
                ],
                evidence=self._flatten_round_evidence(evidence)
                + list(self.knowledge.local_evidence(round_id=round_id)),
                hypotheses=self.state.hypotheses,
            )
            self.writer.write_json(
                f"round_{round_id:02d}/structured_kg_post_validation.json",
                {
                    "report": post_structured_result.report,
                    "entity_count": len(post_structured_result.snapshot.entities),
                    "relation_count": len(post_structured_result.snapshot.relations),
                    "snapshot_id": self.knowledge.structured_sink.last_snapshot_id,
                    "validation_record_count": len(self.validation_records),
                },
            )
            self.writer.event(
                "structured_kg_built",
                {
                    "round_id": round_id,
                    "stage": "post_validation",
                    "entity_count": len(post_structured_result.snapshot.entities),
                    "relation_count": len(post_structured_result.snapshot.relations),
                },
            )

        final_metrics = None
        if rounds_aborted == 0:
            self._progress(
                "final_fit_started",
                "fitting final predictor on all revealed observations",
                phase=CampaignPhase.MODEL_FIT,
                n_train=len(observed_variants),
                model=self.config.model.name,
            )
            final_predictor = self.predictor_factory(
                self.config.model, seed=self.config.seed + self.config.rounds + 1
            )
            self._fit_predictor(final_predictor, observed_variants)
            final_test_variants = self._final_test_variants()
            self._progress(
                "final_predict_started",
                f"predicting {len(final_test_variants)} held-out final-test variants",
                phase=CampaignPhase.PREDICTING,
                n_candidates=len(final_test_variants),
            )
            final_predictions = final_predictor.predict(final_test_variants)
            final_observations = self.backend.open_final_test()
            if {item.variant_id for item in final_test_variants} != {
                item.variant_id for item in final_observations
            }:
                raise RuntimeError("Final-test inputs and oracle labels have different variant IDs")
            self.state.final_test_opened = True
            final_metrics = prediction_metrics(
                final_predictions,
                final_observations,
                metrics=self.config.evaluation.metrics,
                top_k=self.config.evaluation.top_k,
            )
        else:
            self._progress(
                None,
                "skipping final-test after round abort",
                persist=False,
            )

        output_paths = write_campaign_outputs(
            self.writer,
            output=self.config.output,
            wild_type=wild_type,
            state=self.state,
            variants=public_by_id,
            round_metrics=round_metrics,
            validation_records=self.validation_records,
        )
        if getattr(self.critic_agent, "fallback_count", 0):
            self._fallback_nodes.append("batch_critic")
        required_node_failures = tuple(dict.fromkeys(self._required_node_failures))
        fallback_nodes = tuple(dict.fromkeys(self._fallback_nodes))
        completed_rounds = len(round_metrics)
        run_completed = (
            rounds_aborted == 0
            and completed_rounds == self.config.rounds
            and not required_node_failures
        )
        pass_eligible = run_completed and not fallback_nodes
        completion_manifest = CompletionManifest(
            artifact_finalized=True,
            run_status="completed" if run_completed else "failed",
            experiment_status="completed" if run_completed else "failed",
            evaluation_status="eligible" if pass_eligible else "not_evaluated",
            pass_eligible=pass_eligible,
            expected_rounds=self.config.rounds,
            completed_rounds=completed_rounds,
            aborted_rounds=rounds_aborted,
            required_node_failures=required_node_failures,
            fallback_nodes=fallback_nodes,
        )
        summary = {
            "run_id": self.run_id,
            "mode": self.config.mode,
            "condition": self.config.condition or self.config.mode,
            "run_label": self.config.run_label,
            "seed": self.config.seed,
            "round_metrics": round_metrics,
            "final_prediction_metrics": final_metrics,
            "queries_used": len(self.state.selections),
            "hypotheses_generated": len(self.state.hypotheses),
            "selection_records": len(self.state.selections),
            "critique_decisions": len(self.state.critique_decisions),
            "hypothesis_assessments": len(self.state.hypothesis_assessments),
            "rethink_reflections": len(self.state.rethink_reflections),
            "validation_records": len(self.validation_records),
            "selection_driver": selection_driver,
            "fitness_predictors_used_for_generation": (
                self._fitness_predictors_used_for_generation(selection_driver)
            ),
            "active_learning_module": (
                self.config.active_learning.module
                if selection_driver == "active_learning"
                else None
            ),
            "rounds_aborted": rounds_aborted,
            "artifact_finalized": completion_manifest.artifact_finalized,
            "run_status": completion_manifest.run_status,
            "experiment_status": completion_manifest.experiment_status,
            "evaluation_status": completion_manifest.evaluation_status,
            "pass_eligible": completion_manifest.pass_eligible,
            "required_node_failures": list(required_node_failures),
            "fallback_nodes": list(fallback_nodes),
            "finalized": True,
            "data_source": self._data_source_record,
            "run_dir": str(self.writer.run_dir),
            "output_artifacts": output_paths,
        }
        self.writer.write_json("state.json", self.state.as_dict())
        self.writer.write_json("knowledge_graph_edges.json", self.knowledge.graph.export_edges())
        self.writer.write_json("validation_records.json", self.validation_records)
        self.writer.write_json(
            "knowledge_graph_queries.json",
            self.knowledge.graph.export_agent_queries(),
        )
        self.writer.write_json(
            "completion_manifest.json", completion_manifest.model_dump(mode="json")
        )
        self.writer.write_json("summary.json", summary)
        self._progress(
            None,
            "campaign finalized",
            phase=CampaignPhase.FINALIZED,
            persist=False,
        )
        self.writer.event("campaign_finalized", summary)
        self.knowledge.close()
        return summary


def run_campaign(config: ExperimentConfig) -> dict[str, Any]:
    if config.designer.space == "open_design":
        from .open_design import run_open_design

        return run_open_design(config)
    return CampaignRunner(config).run()
