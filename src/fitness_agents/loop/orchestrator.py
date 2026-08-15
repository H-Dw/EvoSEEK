from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import numpy as np

from fitness_agents.acquisition import create_policy
from fitness_agents.agents.critic import CriticAgent, OpenAICriticClient, RuleBasedCriticClient
from fitness_agents.agents.llm import create_llm_client
from fitness_agents.agents.scientist import ScientistAgent
from fitness_agents.config import ExperimentConfig
from fitness_agents.contracts.schemas import (
    CampaignPhase,
    CampaignState,
    Prediction,
    SelectionRecord,
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
from fitness_agents.knowledge import KnowledgeEngine
from fitness_agents.models import create_predictor
from fitness_agents.mutation import create_candidate_generator
from fitness_agents.utils import JsonArtifactWriter, seed_everything
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
        )
        graph_tool = (
            self.knowledge.agent_tool()
            if config.knowledge_enabled and config.knowledge.kg and not config.evidence_deletion
            else None
        )
        self.agent = agent or ScientistAgent(
            create_llm_client(config.llm_provider),
            knowledge_graph=graph_tool,
        )
        if critic_agent is None:
            rule_critic = RuleBasedCriticClient()
            if config.critic.mode == "remote" and config.critic.enabled:
                critic_client = OpenAICriticClient(
                    model=config.critic.model,
                    profile=config.critic.profile,
                    temperature=config.critic.temperature,
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
        self.generator = create_candidate_generator(config.mode)
        knowledge_weight = config.knowledge.soft_weight if config.knowledge_enabled else 0.0
        self.policy = create_policy(
            config.acquisition,
            beta=config.ucb_beta,
            knowledge_weight=knowledge_weight,
        )
        self.state = CampaignState(run_id=self.run_id, mode=config.mode, seed=config.seed)

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
            "llm_provider": self.config.llm_provider,
            "critic": {
                "enabled": self.config.critic.enabled,
                "mode": self.config.critic.mode,
                "provider": self.config.critic.provider,
                "model": self.config.critic.model,
                "profile": self.config.critic.profile,
                "max_revision_attempts": self.config.critic.max_revision_attempts,
                "fallback_policy": self.config.critic.fallback_policy,
            },
            "score_shuffle": self.config.score_shuffle,
            "evidence_deletion": self.config.evidence_deletion,
            "evidence_prefilter_limit": self.config.evidence_prefilter_limit,
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

        for round_id in range(1, self.config.rounds + 1):
            self.state.round_id = round_id
            predictor = self.predictor_factory(self.config.model, seed=self.config.seed + round_id)
            self._fit_predictor(predictor, observed_variants)
            self.state.phase = CampaignPhase.MODEL_FIT
            original_predictions = predictor.predict(remaining)
            working_predictions = (
                _shuffle_prediction_scores(original_predictions, self.rng)
                if self.config.score_shuffle
                else original_predictions
            )
            prediction_by_id = {item.variant_id: item for item in original_predictions}
            working_by_id = {item.variant_id: item for item in working_predictions}

            evidence_targets = remaining
            if (
                self.config.knowledge_enabled
                and self.config.evidence_prefilter_limit > 0
                and len(remaining) > self.config.evidence_prefilter_limit
            ):
                evidence_targets = sorted(
                    remaining,
                    key=lambda item: (
                        working_by_id[item.variant_id].fitness_mean
                        + self.config.ucb_beta * working_by_id[item.variant_id].fitness_std,
                        item.variant_id,
                    ),
                    reverse=True,
                )[: self.config.evidence_prefilter_limit]
                self.writer.event(
                    "evidence_prefilter_applied",
                    {
                        "round_id": round_id,
                        "input_candidates": len(remaining),
                        "evidence_candidates": len(evidence_targets),
                        "criterion": "predictor UCB without oracle labels",
                    },
                )
            evidence = (
                self.knowledge.evidence_for(
                    evidence_targets,
                    round_id=round_id,
                    delete_evidence=self.config.evidence_deletion,
                )
                if self.config.knowledge_enabled
                else {}
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
                    evidence_targets,
                    [working_by_id[item.variant_id] for item in evidence_targets],
                    evidence,
                    round_id=round_id,
                    intervention_tags=inference_interventions,
                )
            hypothesis = None
            if self.config.mode in {"llm_agent", "knowledge_agent"}:
                hypothesis = self.agent.propose_hypothesis(
                    self.state,
                    observed_variants,
                    self.state.observed,
                    _flatten_evidence(evidence),
                )
                self.state.hypotheses.append(hypothesis)
                self.knowledge.graph.add_hypothesis(
                    hypothesis.hypothesis_id,
                    round_id,
                    hypothesis.statement,
                    hypothesis.evidence_ids,
                )
                self.writer.event("hypothesis_proposed", hypothesis.__dict__)
                if self.agent.last_knowledge_query_id is not None:
                    self.writer.event(
                        "knowledge_graph_queried",
                        {
                            "round_id": round_id,
                            "query_id": self.agent.last_knowledge_query_id,
                            "operation": "hypothesis_context",
                        },
                    )

            eligible = self.generator.generate(
                remaining,
                self.state,
                hypothesis,
                evidence,
                self.config.candidate_limit,
            )
            self.state.phase = CampaignPhase.PROPOSED
            knowledge_scores = self.knowledge.scores(evidence) if self.config.knowledge_enabled else {}
            all_scores = self.policy.score(working_predictions, knowledge_scores, self.rng)
            selected_ids = self.policy.select(
                eligible,
                [working_by_id[item.variant_id] for item in eligible],
                all_scores,
                min(self.config.budget_per_round, len(eligible)),
                self.config.diversity_lambda,
            )
            if len(selected_ids) != min(self.config.budget_per_round, len(eligible)):
                raise RuntimeError("Acquisition returned an incomplete batch")

            expected_batch_size = min(self.config.budget_per_round, len(eligible))
            initial_selected_ids = tuple(selected_ids)
            draft_context = {
                "initial_selected_ids": initial_selected_ids,
                "eligible": eligible,
                "working_by_id": working_by_id,
                "all_scores": all_scores,
                "expected_batch_size": expected_batch_size,
                "predictor": predictor,
                "prediction_by_id": prediction_by_id,
                "round_id": round_id,
                "evidence": evidence,
                "hypothesis": hypothesis,
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
                refreshed_predictions = _context["predictor"].predict(candidate_variants)
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
                )

            def record_review_attempt(
                draft, report, decision, _round_id: int = round_id
            ) -> None:
                self.state.phase = CampaignPhase.CRITIQUE_REQUESTED
                self.state.critique_decisions.append(decision)
                folder = f"round_{_round_id:02d}"
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
                if decision.verdict.value == "REVISE":
                    self.state.phase = CampaignPhase.REVISION_REQUESTED
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
                    self.state.phase = CampaignPhase.ROUND_ABORTED
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
            self.state.phase = CampaignPhase.APPROVED
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
                        reason=review_result.decision.summary,
                        intervention_tags=intervention_tags,
                    )
                )
            self.state.selections.extend(records)
            self.state.phase = CampaignPhase.SELECTED
            self.writer.write_selection(round_id, records)
            self.writer.event(
                "batch_selected",
                {
                    "round_id": round_id,
                    "records": records,
                    "global_rank_definition": (
                        "model_rank_all ranks original predictor mean over every unobserved oracle-pool "
                        "candidate before agent filtering; acquisition_rank_all ranks the active policy"
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
            self.state.phase = CampaignPhase.SUBMITTED
            revealed = self.backend.collect(experiment_run_id)
            selected_variants = [public_by_id[variant_id] for variant_id in selected_ids]
            self.state.observed.extend(revealed)
            observed_variants.extend(selected_variants)
            self.state.revealed_variant_ids.update(selected_ids)
            self.knowledge.update(selected_variants, revealed)
            selected_set = set(selected_ids)
            remaining = [item for item in remaining if item.variant_id not in selected_set]
            self.state.phase = CampaignPhase.MEASURED
            metrics = loop_round_metrics(
                self.state.observed,
                revealed,
                total_pool_size=len(original_predictions),
                selected_model_ranks=[record.model_rank_all for record in records],
            )
            metrics["round_id"] = float(round_id)
            round_metrics.append(metrics)
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
                self.state.phase = CampaignPhase.HYPOTHESIS_EVALUATED
                self.writer.write_json(
                    f"round_{round_id:02d}/hypothesis_assessment.json", assessment
                )
                self.writer.event("hypothesis_assessed", assessment.__dict__)

        final_predictor = self.predictor_factory(
            self.config.model, seed=self.config.seed + self.config.rounds + 1
        )
        self._fit_predictor(final_predictor, observed_variants)
        final_test_variants = self._final_test_variants()
        final_predictions = final_predictor.predict(final_test_variants)
        final_observations = self.backend.open_final_test()
        if {item.variant_id for item in final_test_variants} != {
            item.variant_id for item in final_observations
        }:
            raise RuntimeError("Final-test inputs and oracle labels have different variant IDs")
        self.state.final_test_opened = True
        self.state.phase = CampaignPhase.FINALIZED
        final_metrics = prediction_metrics(final_predictions, final_observations)
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
            "rounds_aborted": rounds_aborted,
            "finalized": True,
            "data_source": self._data_source_record,
            "run_dir": str(self.writer.run_dir),
        }
        self.writer.write_json("state.json", self.state.as_dict())
        self.writer.write_json("knowledge_graph_edges.json", self.knowledge.graph.export_edges())
        self.writer.write_json(
            "knowledge_graph_queries.json",
            self.knowledge.graph.export_agent_queries(),
        )
        self.writer.write_json("summary.json", summary)
        self.writer.event("campaign_finalized", summary)
        self.knowledge.close()
        return summary


def run_campaign(config: ExperimentConfig) -> dict[str, Any]:
    return CampaignRunner(config).run()
