from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import numpy as np

from fitness_agents.acquisition import create_policy
from fitness_agents.agents.critic import CriticAgent, OpenAICriticClient, RuleBasedCriticClient
from fitness_agents.agents.llm import create_llm_client
from fitness_agents.agents.rethink import create_rethink_client
from fitness_agents.agents.scientist import ScientistAgent
from fitness_agents.config import ExperimentConfig
from fitness_agents.contracts.schemas import (
    CampaignPhase,
    CampaignState,
    DesignScore,
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
    ExplainVariantOperator,
    HypothesisContextOperator,
    InteractionAblationConfig,
    KGInteractionController,
    KGQueryContext,
    KGQueryPlan,
    KGQueryStep,
    QueryIntent,
)
from fitness_agents.knowledge import KnowledgeEngine
from fitness_agents.models import create_predictor
from fitness_agents.mutation import AgentUncertaintySelector, create_candidate_generator
from fitness_agents.plugin_registry import PluginRegistry
from fitness_agents.reporting import write_campaign_outputs
from fitness_agents.utils import JsonArtifactWriter, bind_progress, reset_progress, seed_everything
from fitness_agents.validation.batch import ApprovalGateway, BatchHardValidator, build_draft_batch

from .backends import ApprovalEnforcingBackend, CsvOracleBackend
from .review import BoundedReviewLoop, ReviewRejected


def _descending_ranks(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda key: (values[key], key), reverse=True)
    return {variant_id: rank for rank, variant_id in enumerate(ordered, start=1)}


def _flatten_evidence(evidence: dict[str, list[Any]], limit: int = 120) -> list[Any]:
    entries = [item for bundle in evidence.values() for item in bundle]
    return sorted(
        entries,
        key=lambda item: (item.confidence * abs(item.score), item.evidence_id),
        reverse=True,
    )[:limit]


def _shuffle_prediction_scores(
    predictions: Sequence[Prediction], rng: np.random.Generator
) -> list[Prediction]:
    means = np.asarray([prediction.fitness_mean for prediction in predictions])
    shuffled = rng.permutation(means)
    return [replace(prediction, fitness_mean=float(shuffled[index])) for index, prediction in enumerate(predictions)]


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
        self.knowledge = KnowledgeEngine(
            config.knowledge,
            graph_path=self.writer.run_dir / "knowledge_graph.sqlite",
            assay_id=config.task.assay_id,
            protein_id=config.task.protein_id,
            validation_config=config.validation,
            structured_graph_path=self.writer.run_dir / "structured_kg.sqlite",
        )
        graph_tool = (
            self.knowledge.agent_tool()
            if config.knowledge_enabled and config.knowledge.kg and not config.evidence_deletion
            else None
        )
        self.agent = agent or ScientistAgent(
            create_llm_client(
                self.config.llm.provider,
                profile=self.config.llm.profile,
                model=self.config.llm.model,
                base_url=self.config.llm.base_url,
                api_key=self.config.llm.api_key,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                reasoning_effort=self.config.llm.reasoning_effort,
                thinking=self.config.llm.thinking,
            ),
            knowledge_graph=graph_tool,
        )
        self.kg_interaction = None
        if graph_tool is not None and config.kg_interaction.enabled:
            operators = PluginRegistry("query_operator")
            operators.register("hypothesis_context", HypothesisContextOperator(graph_tool))
            operators.register("explain_variant", ExplainVariantOperator(graph_tool))
            operators.register("compare_variants", CompareVariantsOperator(graph_tool))
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
        self.rethink_client = create_rethink_client(
            self.config.llm.provider,
            model=self.config.llm.model,
            base_url=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            reasoning_effort=self.config.llm.reasoning_effort,
            thinking=self.config.llm.thinking,
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
                )
                critic_agent = CriticAgent(
                    critic_client,
                    max_retries=config.critic.max_model_retries,
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
        self.generator = create_candidate_generator(config.mode)
        self.agent_selector = AgentUncertaintySelector(config.generation)
        knowledge_weight = config.knowledge.soft_weight if config.knowledge_enabled else 0.0
        self.policy = create_policy(
            config.acquisition,
            beta=config.ucb_beta,
            knowledge_weight=knowledge_weight,
        )
        self.state = CampaignState(run_id=self.run_id, mode=config.mode, seed=config.seed)
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
        steps = [
            KGQueryStep(
                "context",
                "hypothesis_context",
                QueryIntent.CONTEXT,
                {"limit": self.config.kg_interaction.max_rows},
            )
        ]
        if representative_ids:
            steps.append(
                KGQueryStep(
                    "explain",
                    "explain_variant",
                    QueryIntent.EXPLAIN,
                    {"variant_id": representative_ids[0]},
                    ("context",),
                )
            )
        if len(representative_ids) >= 2:
            steps.append(
                KGQueryStep(
                    "compare",
                    "compare_variants",
                    QueryIntent.COMPARE,
                    {"variant_ids": representative_ids},
                    ("context",),
                )
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

    def _config_record(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
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
            "llm_provider": self.config.llm.provider,
            "llm": {
                "provider": self.config.llm.provider,
                "profile": self.config.llm.profile,
                "profile_sha256": getattr(
                    self.agent.client, "profile_sha256", None
                ),
                "model": self.config.llm.model,
                "base_url": self.config.llm.base_url,
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
                "reasoning_effort": self.config.llm.reasoning_effort,
                "thinking": self.config.llm.thinking,
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
                "fallback_policy": self.config.critic.fallback_policy,
                "api_key_ref": self.config.critic.api_key
                if isinstance(self.config.critic.api_key, str)
                and self.config.critic.api_key.startswith("env:")
                else ("configured" if self.config.critic.api_key else None),
            },
            "score_shuffle": self.config.score_shuffle,
            "evidence_deletion": self.config.evidence_deletion,
            "evidence_prefilter_limit": self.config.evidence_prefilter_limit,
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
            },
            "validation": {
                "enabled": self.config.validation.enabled,
                "primary_model": self.config.model.name,
                "additional_models": [item.name for item in self.config.validation.predictor_models],
                "wet_weight": self.config.validation.wet_weight,
                "dry_weight_cap": self.config.validation.dry_weight_cap,
                "recency_decay": self.config.validation.recency_decay,
                "rethink_enabled": self.config.validation.rethink_enabled,
            },
            "kg_interaction": {
                "enabled": self.kg_interaction is not None,
                "enabled_operators": list(self.config.kg_interaction.enabled_operators),
                "max_tool_calls": self.config.kg_interaction.max_tool_calls,
                "stop_when_sufficient": self.config.kg_interaction.stop_when_sufficient,
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
            for variant in (
                self.bundle.initial_variants
                + self.bundle.oracle_pool
            )
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
            if self.config.knowledge_enabled:
                seen_ids: set[str] = set()
                evidence_targets: list[Any] = []
                for item in (*observed_variants, *remaining):
                    if item.variant_id in seen_ids:
                        continue
                    seen_ids.add(item.variant_id)
                    evidence_targets.append(item)
                self._progress(
                    "evidence_started",
                    (
                        f"round {round_id}/{self.config.rounds} scoring evidence for "
                        f"{len(evidence_targets)} variants"
                    ),
                    n_candidates=len(evidence_targets),
                )
                evidence = self.knowledge.evidence_for(
                    evidence_targets,
                    round_id=round_id,
                    delete_evidence=self.config.evidence_deletion,
                )
                if evidence:
                    self.knowledge.graph.add_variants(evidence_targets)
                    self.knowledge.graph.add_evidence(
                        [item for bundle in evidence.values() for item in bundle]
                    )
                self._progress(
                    "evidence_completed",
                    f"round {round_id}/{self.config.rounds} evidence ready",
                    n_candidates=len(evidence_targets),
                )

            structured_variants = list(
                {item.variant_id: item for item in (*observed_variants, *remaining)}.values()
            )
            structured_result = self.knowledge.sync_structured_kg(
                run_id=self.run_id,
                round_id=round_id,
                variants=structured_variants,
                observations=self.state.observed,
                evidence=_flatten_evidence(evidence, limit=max(len(structured_variants) * 4, 120)),
                hypotheses=self.state.hypotheses,
            )
            self.writer.write_json(
                f"round_{round_id:02d}/structured_kg_pre_design.json",
                {
                    "report": structured_result.report,
                    "entity_count": len(structured_result.snapshot.entities),
                    "relation_count": len(structured_result.snapshot.relations),
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
                self.writer.write_json(
                    f"round_{round_id:02d}/kg_interaction.json",
                    interaction_result,
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
                    },
                )

            hypothesis = None
            if self.config.mode in {"llm_agent", "knowledge_agent"}:
                self._progress(
                    "hypothesis_generation_started",
                    f"round {round_id}/{self.config.rounds} requesting scientist hypothesis",
                    phase=CampaignPhase.LLM_HYPOTHESIS,
                    model=self.config.llm.model or self.config.llm.provider,
                )
                hypothesis = self.agent.propose_hypothesis(
                    self.state,
                    observed_variants,
                    self.state.observed,
                    _flatten_evidence(evidence),
                    kg_interaction=interaction_result,
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
                        generation_prediction_sets.append(
                            generation_predictor.predict(eligible)
                        )
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
                    design_predictions = _shuffle_prediction_scores(
                        design_predictions, self.rng
                    )
                all_scores = {
                    item.variant_id: item.fitness_mean for item in design_predictions
                }
                working_by_id = {
                    item.variant_id: item for item in design_predictions
                }
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
                working_by_id = {
                    item.variant_id: item for item in design_predictions
                }
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
                all_scores = self.policy.score(
                    working_predictions, knowledge_scores, self.rng
                )
                working_by_id = {
                    item.variant_id: item for item in working_predictions
                }
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
            self.writer.write_json(
                f"round_{round_id:02d}/design_scores.json", design_scores
            )
            self._progress(
                "design_scored",
                f"round {round_id}/{self.config.rounds} design utilities ready via {selection_driver}",
                phase=CampaignPhase.DESIGN_SCORED,
                n_candidates=len(eligible),
            )
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
                        self.config.generation.use_fitness_predictors
                        if selection_driver == "agent_uq"
                        else selection_driver == "predictor"
                    ),
                },
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
                original_predictions = [
                    working_by_id[item.variant_id] for item in predict_targets
                ]
            prediction_by_id = {
                item.variant_id: item for item in original_predictions
            }
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
                "predictor": validation_predictors[0] if validation_predictors else None,
                "prediction_by_id": prediction_by_id,
                "round_id": round_id,
                "evidence": evidence,
                "hypothesis": hypothesis,
                "rationale_claims": {
                    item.variant_id: item.reason for item in design_scores
                },
            }

            def draft_builder(
                review_attempt: int,
                parent_draft_batch_id: str | None,
                exclusions: set[str],
                _context: dict[str, Any] = draft_context,
            ):
                if review_attempt == 0:
                    candidate_ids = list(_context["initial_selected_ids"])
                else:
                    revised_eligible = [
                        item
                        for item in _context["eligible"]
                        if item.variant_id not in exclusions
                    ]
                    candidate_ids = self.policy.select(
                        revised_eligible,
                        [
                            _context["working_by_id"][item.variant_id]
                            for item in revised_eligible
                        ],
                        _context["all_scores"],
                        min(_context["expected_batch_size"], len(revised_eligible)),
                        self.config.diversity_lambda,
                    )
                candidate_variants = [public_by_id[item] for item in candidate_ids]
                missing_predictions = [
                    item
                    for item in candidate_variants
                    if item.variant_id not in _context["prediction_by_id"]
                ]
                if missing_predictions and _context["predictor"] is not None:
                    refreshed_predictions = _context["predictor"].predict(
                        missing_predictions
                    )
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
                        _context["hypothesis"].hypothesis_id
                        if _context["hypothesis"]
                        else None
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

            def record_review_attempt(
                draft, report, decision, _round_id: int = round_id
            ) -> None:
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
                    on_attempt=record_review_attempt,
                    on_attempt_start=record_review_start,
                )
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
                            "decision_ids": [
                                item.decision_id for item in error.decisions
                            ],
                        },
                    )
                    break

            approved_batch = review_result.approved_batch
            selected_ids = list(approved_batch.candidate_ids)
            self.state.approved_batch_ids.append(approved_batch.draft_batch_id)
            self._progress(
                None,
                f"round {round_id} batch approved ({len(selected_ids)} variants)",
                phase=CampaignPhase.APPROVED,
                persist=False,
            )
            self.writer.write_json(
                f"round_{round_id:02d}/approved_batch.json", approved_batch
            )
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
            rethink_context = {
                "run_id": self.run_id,
                "round_id": round_id,
                "visible_baseline": pre_round_visible_baseline,
                "candidates": [
                    {
                        "variant_id": variant_id,
                        "mutation_notation": public_by_id[variant_id].mutation_notation,
                        "agent_reason": selection_by_id[variant_id].reason,
                        "hypothesis": hypothesis.statement if hypothesis else None,
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
            reflections = ()
            if self.config.validation.rethink_enabled:
                try:
                    reflections = self.rethink_client.reflect_round(
                        context=rethink_context
                    )
                    if {item.variant_id for item in reflections} != set(selected_ids):
                        raise ValueError("ReThink output did not cover every selected variant")
                except Exception as error:  # noqa: BLE001 - provider boundary must degrade safely
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
                            reflection_id=(
                                reflection.reflection_id if reflection else None
                            ),
                            reflection_verdict=(
                                reflection.verdict if reflection else None
                            ),
                            reflection_summary=(
                                reflection.summary if reflection else ""
                            ),
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
                variants=list(
                    {
                        item.variant_id: item
                        for item in (*observed_variants, *remaining, *predict_targets)
                    }.values()
                ),
                observations=self.state.observed,
                predictions=[
                    prediction
                    for predictions in validation_prediction_sets
                    for prediction in predictions
                ],
                evidence=_flatten_evidence(
                    evidence,
                    limit=max(len(evidence) * 4, 120),
                ),
                hypotheses=self.state.hypotheses,
            )
            self.writer.write_json(
                f"round_{round_id:02d}/structured_kg_post_validation.json",
                {
                    "report": post_structured_result.report,
                    "entity_count": len(post_structured_result.snapshot.entities),
                    "relation_count": len(post_structured_result.snapshot.relations),
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
        summary = {
            "run_id": self.run_id,
            "mode": self.config.mode,
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
                self.config.generation.use_fitness_predictors
                if selection_driver == "agent_uq"
                else selection_driver == "predictor"
            ),
            "rounds_aborted": rounds_aborted,
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
    return CampaignRunner(config).run()
