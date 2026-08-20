#!/usr/bin/env python3
"""Run hierarchical Scientist and base-KG ablations across folds of one task.

Same-task folds are isolated processes. Default matrix is 4 conditions x 3 folds
= 12 jobs; --max-parallel 4 completes them in three waves. Hierarchical jobs fan
out 3 child LLM calls; kg_base* jobs do not. Use --max-parallel 2 if DeepSeek
contends. RAG jobs share a prebuilt read-only corpus and use per-condition/fold
overlay databases so parallel workers do not share a writer. Parallel Kermut
workers are pinned to distinct GPUs with --cuda-devices (default auto).

Formal runs call DeepSeek, use Qwen for RAG, and keep Kermut acquisition and
post-selection validation roles explicit. Placeholder predictors are not accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fitness_agents.agents.output_guards import MAX_OUTPUT_TOKENS
from fitness_agents.agents.remote_llm import load_project_env, resolve_secret
from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.contracts.capabilities import PredictorCapabilities
from fitness_agents.contracts.schemas import Prediction
from fitness_agents.models import available_predictors
from fitness_agents.reporting import aggregate_runs
from fitness_agents.utils.cuda_jobs import (
    CudaDevicePool,
    cuda_assignment_record,
    environment_with_cuda_device,
    parse_cuda_devices_arg,
    resolve_cuda_device_pool,
)

REQUIRED_CHANNELS = ("physchem", "conservation", "structure")
FEATURE_OPERATORS = frozenset(
    {
        "query_physchem_delta",
        "query_evolutionary_profile",
        "query_structure_environment",
        "query_feature_bundle",
    }
)
BASE_AUDIT_ITEMS = ("MutationEffectEstimate", "HAS_MUTATION", "ABOUT_MUTATION")
SUMMARY_KEYS = {
    "run_id",
    "mode",
    "seed",
    "round_metrics",
    "final_prediction_metrics",
    "queries_used",
    "data_source",
}


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    hierarchical: bool
    rag: bool
    channels: tuple[str, ...]
    active_learning: bool
    test_goal: str


CONDITION_SPECS: dict[str, ConditionSpec] = {
    "hierarchical": ConditionSpec(
        condition_id="hierarchical",
        hierarchical=True,
        rag=False,
        channels=REQUIRED_CHANNELS,
        active_learning=False,
        test_goal="Three-channel hierarchical Scientist with base KG and no RAG.",
    ),
    "single": ConditionSpec(
        condition_id="single",
        hierarchical=False,
        rag=False,
        channels=REQUIRED_CHANNELS,
        active_learning=False,
        test_goal="Single Scientist with three feature channels; hierarchy off.",
    ),
    "kg_base": ConditionSpec(
        condition_id="kg_base",
        hierarchical=False,
        rag=False,
        channels=(),
        active_learning=False,
        test_goal="Base observation KG without RAG or three-channel feature tools.",
    ),
    "kg_base_rag": ConditionSpec(
        condition_id="kg_base_rag",
        hierarchical=False,
        rag=True,
        channels=(),
        active_learning=False,
        test_goal="Base observation KG plus document RAG; no three-channel feature tools.",
    ),
    "kg_base_al": ConditionSpec(
        condition_id="kg_base_al",
        hierarchical=False,
        rag=False,
        channels=(),
        active_learning=True,
        test_goal="Base observation KG without RAG, with active-learning acquisition.",
    ),
    "kg_3features_rag": ConditionSpec(
        condition_id="kg_3features_rag",
        hierarchical=True,
        rag=True,
        channels=REQUIRED_CHANNELS,
        active_learning=False,
        test_goal="Three-channel hierarchical Scientist with document RAG.",
    ),
}
ALLOWED_CONDITIONS = tuple(CONDITION_SPECS)
DEFAULT_CONDITIONS = ("kg_base", "kg_base_rag", "kg_base_al", "kg_3features_rag")
FORMAL_MAX_TOKENS = MAX_OUTPUT_TOKENS
FORMAL_DEFAULT_MAX_TOKENS = 32768
_RAG_LOCAL_TEMPLATE = None


class _CanaryPlaceholderPredictor:
    """Deterministic no-training predictor for API/prompt canaries only."""

    capabilities = PredictorCapabilities()

    def __init__(self, *, seed: int) -> None:
        self.seed = seed
        self.model_version = f"placeholder-canary-sha256-seed{seed}"

    def fit(self, *_args: Any, **_kwargs: Any) -> _CanaryPlaceholderPredictor:
        return self

    def predict(self, variants: Any) -> list[Prediction]:
        predictions = []
        for variant in variants:
            digest = hashlib.sha256(
                f"{self.seed}|{variant.variant_id}".encode()
            ).digest()
            mean = (int.from_bytes(digest[:8], "big") / float(2**64 - 1)) * 2.0 - 1.0
            std = 0.25
            predictions.append(
                Prediction(
                    variant_id=variant.variant_id,
                    fitness_mean=mean,
                    fitness_std=std,
                    interval_90=(mean - 1.645 * std, mean + 1.645 * std),
                    ood_score=0.5,
                    component_scores={"placeholder_hash": mean},
                    model_version=self.model_version,
                )
            )
        return predictions


def _canary_placeholder_predictor_factory(_config: Any, *, seed: int) -> Any:
    return _CanaryPlaceholderPredictor(seed=seed)


@dataclass(frozen=True)
class HierarchicalJob:
    index: int
    condition: str
    fold_index: int
    seed: int
    expected_rounds: int
    expected_budget: int
    expected_candidate_limit: int
    command: tuple[str, ...]


@dataclass(frozen=True)
class HierarchicalJobResult:
    index: int
    condition: str
    fold_index: int
    seed: int
    status: str
    returncode: int
    stdout_log: str
    stderr_log: str
    summary: dict[str, Any] | None = None
    audit: dict[str, Any] | None = None
    error: str | None = None
    cuda_device: str | None = None


def _parse_folds(value: str, n_folds: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(n_folds))
    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not folds:
        raise ValueError("At least one fold must be selected")
    if len(folds) != len(set(folds)):
        raise ValueError("Fold selection contains duplicates")
    invalid = [fold for fold in folds if fold < 0 or fold >= n_folds]
    if invalid:
        raise ValueError(f"Fold indices outside [0, {n_folds - 1}]: {invalid}")
    return folds


def _parse_conditions(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("At least one condition must be selected")
    if len(items) != len(set(items)):
        raise ValueError("Condition selection contains duplicates")
    unknown = [item for item in items if item not in ALLOWED_CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    return items


def _validate_max_tokens(value: int) -> int:
    if not 1 <= value <= FORMAL_MAX_TOKENS:
        raise ValueError(f"max_tokens must be between 1 and {FORMAL_MAX_TOKENS}")
    return value


def apply_token_budget(config: Any, max_tokens: int) -> Any:
    budget = _validate_max_tokens(max_tokens)
    return replace(
        config,
        llm=replace(config.llm, max_tokens=budget),
        critic=replace(config.critic, max_tokens=budget),
        hierarchical_hypothesis=replace(
            config.hierarchical_hypothesis,
            child_max_tokens=budget,
            child_critic_max_tokens=budget,
            main_critic_max_tokens=budget,
        ),
    )


def validate_formal_fitness_configuration(config: Any) -> Any:
    """Fail closed unless the formal matrix keeps predictor roles explicit."""

    if config.model.name != "kermut":
        raise ValueError(
            "Formal hierarchical runs require model_config=configs/model/kermut.yaml; "
            "one-hot predictors are test-only explicit overrides outside this runner"
        )
    if config.model.name not in available_predictors():
        raise ValueError("The configured Kermut predictor is not registered")
    if not config.validation.enabled:
        raise ValueError("Formal hierarchical runs require post-selection dry validation")
    if config.generation.use_fitness_predictors:
        raise ValueError(
            "Formal Agent-UQ routes must not mix fitness predictions into generation"
        )
    if config.generation.predictor_models:
        raise ValueError(
            "Formal Agent-UQ routes must leave generation.predictor_models empty"
        )
    posterior_models = config.active_learning.posterior.predictor_models or (
        config.model,
    )
    invalid_posterior = [item.name for item in posterior_models if item.name != "kermut"]
    if invalid_posterior:
        raise ValueError(
            "The kg_base_al posterior must use Kermut; invalid models: "
            f"{invalid_posterior}"
        )
    return config


def apply_condition(
    config: Any,
    condition: str,
    *,
    fold: int,
    seed: int,
    output_root: Path,
) -> Any:
    spec = CONDITION_SPECS.get(condition)
    if spec is None:
        raise ValueError(f"Unknown condition {condition!r}")
    hierarchical = replace(config.hierarchical_hypothesis, enabled=spec.hierarchical)
    if spec.hierarchical and not config.hierarchical_hypothesis.enabled:
        raise ValueError("hierarchical condition requires hierarchical_hypothesis.enabled")
    local = _local_knowledge_for_spec(config, spec, fold=fold, output_root=output_root)
    if spec.rag:
        embedding = local.retrieval.embedding_api_config
        reranker = local.retrieval.reranker_api_config
        if (
            local.retrieval.embedding_backend != "api"
            or embedding is None
            or embedding.model_family != "qwen"
            or embedding.model != "text-embedding-v4"
            or local.retrieval.reranker_backend != "api"
            or reranker is None
            or reranker.model_family != "qwen"
            or reranker.model != "qwen3-rerank"
        ):
            raise ValueError(
                "Formal RAG conditions require Qwen text-embedding-v4 plus qwen3-rerank"
            )
        if not local.allow_remote_context:
            raise ValueError("Formal RAG conditions must expose bounded RAG context to DeepSeek")
    knowledge = replace(
        config.knowledge,
        physchem="physchem" in spec.channels,
        conservation="conservation" in spec.channels,
        structure="structure" in spec.channels,
        kg=True,
        local_knowledge=local,
    )
    generation = replace(
        config.generation,
        selection_driver="active_learning" if spec.active_learning else "agent_uq",
        quota_allocation=replace(
            config.generation.quota_allocation,
            enabled=not spec.active_learning,
        ),
    )
    active_learning = replace(config.active_learning, enabled=spec.active_learning)
    return replace(
        config,
        seed=seed,
        task=replace(config.task, fold_index=fold),
        knowledge=knowledge,
        kg_interaction=_interaction_for_spec(config, spec),
        generation=generation,
        active_learning=active_learning,
        hierarchical_hypothesis=hierarchical,
        output_root=output_root,
        condition=condition,
        run_label=f"GB1-hierarchical-{condition}-f{fold:02d}",
    )


def _rag_local_knowledge_template() -> Any:
    global _RAG_LOCAL_TEMPLATE
    if _RAG_LOCAL_TEMPLATE is None:
        _RAG_LOCAL_TEMPLATE = load_experiment_config(
            project_root() / "configs/experiments/gb1_reasoning_routes_base.yaml"
        ).knowledge.local_knowledge
    return _RAG_LOCAL_TEMPLATE


def _local_knowledge_for_spec(
    config: Any, spec: ConditionSpec, *, fold: int, output_root: Path
) -> Any:
    if not spec.rag:
        return replace(config.knowledge.local_knowledge, enabled=False)
    template = _rag_local_knowledge_template()
    return replace(
        template,
        enabled=True,
        corpus_mode="read_only_prebuilt",
        retrieval_overlay_path=output_root.parent
        / "local_knowledge"
        / f"{spec.condition_id}-f{fold:02d}-overlay.sqlite",
    )


def require_prebuilt_rag_corpus(conditions: Sequence[str]) -> None:
    if not any(CONDITION_SPECS[item].rag for item in conditions):
        return
    local = _rag_local_knowledge_template()
    corpus_path = local.corpus_index_path or local.index_path
    if corpus_path is None or not corpus_path.is_file():
        raise ValueError(
            "Formal parallel RAG runs require one prebuilt shared Qwen corpus index. "
            "Build it before launching workers with: python -m fitness_agents.cli "
            "knowledge index configs/experiments/gb1_reasoning_routes_base.yaml"
        )
    from fitness_agents.local_knowledge.index import SQLiteLocalKnowledgeIndex

    index = SQLiteLocalKnowledgeIndex(corpus_path, read_only=True)
    try:
        index.prebuilt_report()
    finally:
        index.close()


def required_tool_calls(
    spec: ConditionSpec,
    *,
    variant_limit: int,
    feature_tool_strategy: str = "independent_and_joint",
) -> int:
    count = 2  # hypothesis_context + query_assay_association
    if spec.channels:
        if feature_tool_strategy in {"independent", "independent_and_joint"}:
            count += len(spec.channels) * max(1, variant_limit)
        if feature_tool_strategy in {"joint", "independent_and_joint"}:
            count += max(1, variant_limit)
    count += 1  # truncation audit
    if spec.rag:
        count += 2  # query_local_knowledge + query_structured_claims
    return count + 2  # explain_variant + compare_variants


def _interaction_for_spec(config: Any, spec: ConditionSpec) -> Any:
    if spec.channels:
        operators = tuple(
            item
            for item in config.kg_interaction.enabled_operators
            if item not in {"query_local_knowledge", "query_structured_claims"}
        )
        if spec.rag:
            operators = operators + ("query_local_knowledge", "query_structured_claims")
        return replace(
            config.kg_interaction,
            enabled=True,
            enabled_operators=operators,
            feature_tool_strategy=config.kg_interaction.feature_tool_strategy,
            feature_channels=spec.channels,
            max_tool_calls=max(
                config.kg_interaction.max_tool_calls,
                required_tool_calls(
                    spec,
                    variant_limit=config.kg_interaction.feature_variant_limit,
                    feature_tool_strategy=config.kg_interaction.feature_tool_strategy,
                ),
            ),
        )
    operators = [
        "hypothesis_context",
        "query_assay_association",
        "query_evidence_provenance",
        "query_kg_truncation_audit",
    ]
    if spec.rag:
        operators.extend(("query_local_knowledge", "query_structured_claims"))
    operators.extend(("explain_variant", "compare_variants"))
    return replace(
        config.kg_interaction,
        enabled=True,
        enabled_operators=tuple(operators),
        feature_tool_strategy="context_only",
        feature_channels=("physchem",),
        truncation_audit_items=BASE_AUDIT_ITEMS,
        max_tool_calls=max(
            config.kg_interaction.max_tool_calls,
            required_tool_calls(spec, variant_limit=config.kg_interaction.feature_variant_limit),
        ),
    )


def _check(name: str, passed: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _load_completion_auditor():
    path = Path(__file__).resolve().parent / "audit_agent_completion.py"
    spec = importlib.util.spec_from_file_location(
        "fitness_agents_hierarchical_completion_audit", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_run


def _pipeline_branch_status(pipeline: dict[str, Any]) -> dict[str, str]:
    return {str(item.get("channel")): str(item.get("status")) for item in pipeline.get("branches", ())}


def audit_hierarchical_run(
    summary: dict[str, Any],
    *,
    condition: str,
    expected_fold: int,
    expected_rounds: int,
    expected_budget: int,
    expected_candidate_limit: int,
) -> dict[str, Any]:
    run_dir = Path(summary["run_dir"])
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    completion_audit = _load_completion_auditor()(run_dir)
    round_dirs = sorted(path for path in run_dir.glob("round_*") if path.is_dir())
    completed_rounds = len(summary.get("round_metrics") or ())
    expected_queries = expected_budget * expected_rounds
    hierarchy = config.get("hierarchical_hypothesis") or {}
    required = list(hierarchy.get("required_channels") or ())
    spec = CONDITION_SPECS[condition]
    knowledge_channels = config.get("knowledge_channels") or {}
    interaction_cfg = config.get("kg_interaction") or {}
    runtime = config.get("knowledge_runtime") or {}
    local_runtime = runtime.get("local_knowledge") or {}
    generation_cfg = config.get("generation") or {}
    quota_cfg = generation_cfg.get("quota_allocation") or {}
    al_cfg = config.get("active_learning") or {}
    enabled_operators = set(interaction_cfg.get("enabled_operators") or ())
    checks = [
        _check("summary_has_required_keys", SUMMARY_KEYS.issubset(summary), sorted(SUMMARY_KEYS)),
        _check("completion_manifest_audit", completion_audit.get("passed") is True, completion_audit.get("errors")),
        _check("artifact_finalized", (completion_audit.get("manifest") or {}).get("artifact_finalized") is True),
        _check(
            "run_status_completed",
            (completion_audit.get("manifest") or {}).get("run_status") == "completed",
        ),
        _check("formal_pass_eligible", (completion_audit.get("manifest") or {}).get("pass_eligible") is True),
        _check(
            "planned_actual_batch_sizes_match_protocol",
            (completion_audit.get("manifest") or {}).get("planned_batch_sizes")
            == [expected_budget] * expected_rounds
            and (completion_audit.get("manifest") or {}).get(
                "actual_batch_sizes"
            )
            == [expected_budget] * expected_rounds,
            {
                "planned": (completion_audit.get("manifest") or {}).get(
                    "planned_batch_sizes"
                ),
                "actual": (completion_audit.get("manifest") or {}).get(
                    "actual_batch_sizes"
                ),
            },
        ),
        _check("rounds_aborted_is_zero", int(summary.get("rounds_aborted") or 0) == 0, summary.get("rounds_aborted")),
        _check("condition_matches", summary.get("condition") == condition, summary.get("condition")),
        _check(
            "fold_matches_schedule",
            summary.get("data_source", {}).get("fold_index") == expected_fold,
            summary.get("data_source", {}).get("fold_index"),
        ),
        _check("llm_is_non_mock", config.get("llm_provider") != "mock", config.get("llm_provider")),
        _check(
            "placeholder_predictor_disabled",
            summary.get("placeholder_predictor") is not True,
            summary.get("placeholder_predictor"),
        ),
        _check(
            "candidate_pool_predictor_scope_matches",
            bool(summary.get("fitness_predictors_used_for_generation"))
            == spec.active_learning,
            summary.get("fitness_predictors_used_for_generation"),
        ),
        _check(
            "fitness_model_is_kermut",
            config.get("model") == "kermut",
            config.get("model"),
        ),
        _check(
            "validation_enabled_for_fitness_scoring",
            (config.get("validation") or {}).get("enabled") is True,
            config.get("validation"),
        ),
        _check(
            "llm_max_tokens_is_bounded_formal_budget",
            1 <= int((config.get("llm") or {}).get("max_tokens") or 0) <= FORMAL_MAX_TOKENS,
            (config.get("llm") or {}).get("max_tokens"),
        ),
        _check(
            "critic_max_tokens_is_bounded_formal_budget",
            1 <= int((config.get("critic") or {}).get("max_tokens") or 0) <= FORMAL_MAX_TOKENS,
            (config.get("critic") or {}).get("max_tokens"),
        ),
        _check("rounds_match_protocol", int(config.get("rounds") or 0) == expected_rounds, config.get("rounds")),
        _check(
            "candidate_pool_matches_protocol",
            int(config.get("candidate_limit") or 0) == expected_candidate_limit,
            {
                "actual": config.get("candidate_limit"),
                "expected": expected_candidate_limit,
            },
        ),
        _check(
            "budget_matches_protocol",
            int(config.get("budget_per_round") or 0) == expected_budget,
            config.get("budget_per_round"),
        ),
        _check(
            "completed_rounds_match_protocol",
            completed_rounds == expected_rounds,
            {"completed_rounds": completed_rounds, "expected_rounds": expected_rounds},
        ),
        _check(
            "queries_match_protocol",
            int(summary.get("queries_used") or 0) == expected_queries,
            {
                "queries_used": summary.get("queries_used"),
                "expected_queries": expected_queries,
            },
        ),
    ]
    checks.extend(
        [
            _check("hierarchy_enabled_matches", hierarchy.get("enabled") is spec.hierarchical, hierarchy),
            _check(
                "agent_uq_generation_predictors_disabled",
                generation_cfg.get("use_fitness_predictors") is False,
                generation_cfg.get("use_fitness_predictors"),
            ),
            _check(
                "active_learning_posterior_is_kermut",
                (not spec.active_learning)
                or (al_cfg.get("posterior_models") == ["kermut"]),
                al_cfg.get("posterior_models"),
            ),
            _check("rag_runtime_matches", bool(local_runtime.get("enabled")) == spec.rag, local_runtime),
            _check(
                "rag_allowed_in_scientist_context",
                (not spec.rag) or bool(local_runtime.get("scientist_context_allowed")),
                local_runtime.get("scientist_context_allowed"),
            ),
            _check(
                "selection_driver_matches",
                generation_cfg.get("selection_driver")
                == ("active_learning" if spec.active_learning else "agent_uq"),
                generation_cfg.get("selection_driver"),
            ),
            _check(
                "active_learning_enabled_matches",
                bool(al_cfg.get("enabled")) == spec.active_learning,
                al_cfg.get("enabled"),
            ),
            _check(
                "quota_allocation_matches_driver",
                bool(quota_cfg.get("enabled")) == (not spec.active_learning),
                quota_cfg,
            ),
        ]
    )
    for channel in REQUIRED_CHANNELS:
        expected = channel in spec.channels
        checks.append(
            _check(
                f"knowledge_channel_{channel}_matches",
                bool(knowledge_channels.get(channel)) is expected,
                knowledge_channels.get(channel),
            )
        )
    if spec.channels:
        checks.append(
            _check(
                "feature_tools_present",
                bool(enabled_operators.intersection(FEATURE_OPERATORS)),
                sorted(enabled_operators),
            )
        )
    else:
        checks.append(
            _check(
                "feature_tools_absent",
                not enabled_operators.intersection(FEATURE_OPERATORS),
                sorted(enabled_operators),
            )
        )
        checks.append(
            _check(
                "feature_tool_strategy_is_context_only",
                interaction_cfg.get("feature_tool_strategy") == "context_only",
                interaction_cfg.get("feature_tool_strategy"),
            )
        )
    rag_operators = {"query_local_knowledge", "query_structured_claims"}
    rag_ops_ok = (
        rag_operators.issubset(enabled_operators)
        if spec.rag
        else not rag_operators.intersection(enabled_operators)
    )
    checks.append(
        _check(
            "rag_operators_match",
            rag_ops_ok,
            sorted(enabled_operators.intersection(rag_operators)),
        )
    )
    for round_dir in round_dirs:
        scope_path = round_dir / "prediction_scope_receipt.json"
        checks.append(
            _check(f"{round_dir.name}_prediction_scope_receipt", scope_path.is_file())
        )
        if not scope_path.is_file():
            continue
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        calls = scope.get("dry_validation_calls") or []
        checks.extend(
            [
                _check(
                    f"{round_dir.name}_approved_batch_size",
                    int(scope.get("approved_batch_size") or 0) == expected_budget,
                    scope.get("approved_batch_size"),
                ),
                _check(
                    f"{round_dir.name}_candidate_scoring_bound",
                    int(scope.get("planned_candidate_count") or 0)
                    == expected_candidate_limit
                    and int(scope.get("round_candidate_count") or 0)
                    <= expected_candidate_limit
                    and int(scope.get("acquisition_prediction_count") or 0)
                    <= expected_candidate_limit
                    and scope.get(
                        "acquisition_predictions_within_round_candidate_set"
                    )
                    is True,
                    scope,
                ),
                _check(
                    f"{round_dir.name}_dry_validation_selected_only",
                    scope.get("dry_validation_scope")
                    == "draft_selected_candidates_only"
                    and scope.get("all_dry_validation_targets_were_draft_selected")
                    is True
                    and all(
                        int(item.get("candidate_count") or 0) <= expected_budget
                        for item in calls
                    ),
                    scope,
                ),
                _check(
                    f"{round_dir.name}_acquisition_prediction_scope",
                    scope.get("acquisition_prediction_scope")
                    == ("candidate_pool" if spec.active_learning else "none"),
                    scope.get("acquisition_prediction_scope"),
                ),
                _check(
                    f"{round_dir.name}_oracle_scope",
                    scope.get("oracle_measurement_scope")
                    == "approved_batch_only",
                    scope.get("oracle_measurement_scope"),
                ),
            ]
        )
    if spec.hierarchical:
        checks.append(
            _check(
                "required_channels_are_three_feature_channels",
                tuple(required) == REQUIRED_CHANNELS,
                required,
            )
        )
        checks.append(
            _check(
                "child_role_max_tokens_are_formal_budget",
                all(
                    1 <= int(hierarchy.get(name) or 0) <= FORMAL_MAX_TOKENS
                    for name in (
                        "child_max_tokens",
                        "child_critic_max_tokens",
                        "main_critic_max_tokens",
                    )
                ),
                {
                    "child_max_tokens": hierarchy.get("child_max_tokens"),
                    "child_critic_max_tokens": hierarchy.get("child_critic_max_tokens"),
                    "main_critic_max_tokens": hierarchy.get("main_critic_max_tokens"),
                },
            )
        )
        pipeline_rounds = [
            path for path in round_dirs if (path / "hypothesis_pipeline.json").is_file()
        ]
        checks.append(
            _check(
                "pipeline_present_for_every_completed_round",
                len(pipeline_rounds) == expected_rounds,
                {
                    "pipeline_rounds": len(pipeline_rounds),
                    "expected_rounds": expected_rounds,
                },
            )
        )
        for round_dir in round_dirs:
            pipeline_path = round_dir / "hypothesis_pipeline.json"
            checks.append(_check(f"{round_dir.name}_hypothesis_pipeline", pipeline_path.is_file()))
            if not pipeline_path.is_file():
                continue
            pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
            branch_status = _pipeline_branch_status(pipeline)
            main_review = pipeline.get("main_review") or {}
            checks.append(
                _check(
                    f"{round_dir.name}_pipeline_succeeded",
                    pipeline.get("status") == "SUCCEEDED",
                    pipeline.get("failure_code"),
                )
            )
            checks.append(
                _check(
                    f"{round_dir.name}_three_branches_succeeded",
                    branch_status == {channel: "SUCCEEDED" for channel in REQUIRED_CHANNELS},
                    branch_status,
                )
            )
            checks.append(
                _check(
                    f"{round_dir.name}_main_review_approved",
                    main_review.get("verdict") == "APPROVE",
                    main_review.get("verdict"),
                )
            )
    failed = [item for item in checks if not item["passed"]]
    return {
        "condition": condition,
        "fold_index": expected_fold,
        "run_id": summary.get("run_id"),
        "run_dir": str(run_dir),
        "passed": not failed,
        "checks": checks,
        "failed_checks": [item["name"] for item in failed],
    }


def _build_jobs(
    *,
    script_path: Path,
    config_path: Path,
    conditions: list[str],
    folds: list[int],
    seed: int,
    expected_rounds: int,
    expected_budget: int,
    expected_candidate_limit: int,
    output_root: Path,
    python_executable: str,
    placeholder_predictor: bool = False,
    max_tokens: int = FORMAL_DEFAULT_MAX_TOKENS,
    disable_batch_control_review: bool = False,
    disable_batch_diversity_review: bool = False,
) -> list[HierarchicalJob]:
    jobs: list[HierarchicalJob] = []
    # Fold-major keeps each four-job wave paired on one fold and limits each wave
    # to one hierarchical job, whose three child branches fan out internally.
    for fold in folds:
        for condition in conditions:
            command = (
                python_executable,
                str(script_path),
                "--config",
                str(config_path),
                "--worker-fold",
                str(fold),
                "--worker-condition",
                condition,
                "--seed",
                str(seed),
                "--worker-output-root",
                str(output_root),
                "--worker-rounds",
                str(expected_rounds),
                "--worker-max-tokens",
                str(max_tokens),
                "--worker-candidate-limit",
                str(expected_candidate_limit),
            )
            if placeholder_predictor:
                command = (*command, "--worker-placeholder-predictor")
            if disable_batch_control_review:
                command = (*command, "--disable-batch-control-review")
            if disable_batch_diversity_review:
                command = (*command, "--disable-batch-diversity-review")
            jobs.append(
                HierarchicalJob(
                    index=len(jobs),
                    condition=condition,
                    fold_index=fold,
                    seed=seed,
                    expected_rounds=expected_rounds,
                    expected_budget=expected_budget,
                    expected_candidate_limit=expected_candidate_limit,
                    command=command,
                )
            )
    return jobs


def _run_one_job(
    job: HierarchicalJob,
    *,
    project_dir: Path,
    log_dir: Path,
    timeout_seconds: float | None,
    cuda_pool: CudaDevicePool | None = None,
) -> HierarchicalJobResult:
    prefix = f"{job.condition}-f{job.fold_index:02d}-s{job.seed}"
    stdout_path = log_dir / f"{prefix}.stdout.log"
    stderr_path = log_dir / f"{prefix}.stderr.log"
    cuda_device = cuda_pool.acquire() if cuda_pool is not None else None

    def _write_logs(stdout: str, stderr: str) -> None:
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

    def _result(**kwargs: Any) -> HierarchicalJobResult:
        return HierarchicalJobResult(
            index=job.index,
            condition=job.condition,
            fold_index=job.fold_index,
            seed=job.seed,
            stdout_log=str(stdout_path),
            stderr_log=str(stderr_path),
            cuda_device=cuda_device,
            **kwargs,
        )

    try:
        try:
            completed = subprocess.run(
                list(job.command),
                cwd=project_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
                env=environment_with_cuda_device(cuda_device),
            )
            _write_logs(completed.stdout, completed.stderr)
            if completed.returncode != 0:
                return _result(
                    status="failed",
                    returncode=completed.returncode,
                    error=f"Campaign exited with code {completed.returncode}",
                )
            try:
                summary = json.loads(completed.stdout)
                audit = audit_hierarchical_run(
                    summary,
                    condition=job.condition,
                    expected_fold=job.fold_index,
                    expected_rounds=job.expected_rounds,
                    expected_budget=job.expected_budget,
                    expected_candidate_limit=job.expected_candidate_limit,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as error:
                return _result(
                    status="audit_failed",
                    returncode=0,
                    error=f"{type(error).__name__}: {error}",
                )
            return _result(
                status="passed" if audit["passed"] else "audit_failed",
                returncode=0,
                summary=summary,
                audit=audit,
                error=None if audit["passed"] else f"Audit failed: {audit['failed_checks']}",
            )
        except subprocess.TimeoutExpired as timeout_error:
            stdout = timeout_error.stdout or ""
            stderr = timeout_error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            _write_logs(stdout, stderr)
            return _result(
                status="timeout",
                returncode=124,
                error=f"Campaign exceeded timeout of {timeout_seconds} seconds",
            )
    finally:
        if cuda_pool is not None and cuda_device is not None:
            cuda_pool.release(cuda_device)


def run_fold_jobs(
    jobs: list[HierarchicalJob],
    *,
    max_parallel: int,
    project_dir: Path,
    output_dir: Path,
    timeout_seconds: float | None = None,
    cuda_devices: Sequence[str] | None = None,
) -> list[HierarchicalJobResult]:
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=False)
    results: list[HierarchicalJobResult] = []
    workers = min(max_parallel, len(jobs))
    cuda_pool = CudaDevicePool(cuda_devices) if cuda_devices else None
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hierarchical-fold") as pool:
        futures = {
            pool.submit(
                _run_one_job,
                job,
                project_dir=project_dir,
                log_dir=output_dir,
                timeout_seconds=timeout_seconds,
                cuda_pool=cuda_pool,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            gpu = f" gpu={result.cuda_device}" if result.cuda_device is not None else ""
            print(
                f"condition={result.condition} fold={result.fold_index:02d} "
                f"status={result.status} returncode={result.returncode}{gpu}",
                flush=True,
            )
    return sorted(results, key=lambda item: item.index)


def paired_fold_integrity(
    results: list[HierarchicalJobResult],
    *,
    folds: list[int],
    conditions: list[str],
) -> list[dict[str, Any]]:
    integrity: list[dict[str, Any]] = []
    for fold in folds:
        fold_results = [
            item
            for item in results
            if item.fold_index == fold and item.status == "passed" and item.summary is not None
        ]
        assignment_hashes = {
            item.summary.get("data_source", {}).get("assignment_sha256") for item in fold_results
        }
        assignment_hashes.discard(None)
        actual_conditions = {item.condition for item in fold_results}
        integrity.append(
            {
                "fold_index": fold,
                "passed": (
                    len(fold_results) == len(conditions)
                    and actual_conditions == set(conditions)
                    and len(assignment_hashes) == 1
                ),
                "expected_conditions": list(conditions),
                "actual_conditions": sorted(actual_conditions),
                "assignment_sha256": sorted(str(item) for item in assignment_hashes),
            }
        )
    return integrity


def _worker(args: argparse.Namespace) -> None:
    root = project_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    seed = args.seed
    if seed is None:
        raise SystemExit("Worker requires --seed")
    if args.worker_condition not in ALLOWED_CONDITIONS:
        raise SystemExit(f"Unknown condition {args.worker_condition!r}")
    if args.worker_output_root is None:
        raise SystemExit("Worker requires --worker-output-root")
    from fitness_agents.loop import run_campaign

    config = apply_condition(
        load_experiment_config(config_path),
        args.worker_condition,
        fold=args.worker_fold,
        seed=seed,
        output_root=args.worker_output_root.resolve(),
    )
    try:
        require_prebuilt_rag_corpus([args.worker_condition])
    except ValueError as error:
        raise SystemExit(str(error)) from error
    config = validate_formal_fitness_configuration(config)
    if args.worker_rounds is not None:
        if args.worker_rounds < 1:
            raise SystemExit("Worker rounds must be positive")
        config = replace(config, rounds=args.worker_rounds)
    if args.worker_candidate_limit is not None:
        if args.worker_candidate_limit < config.budget_per_round:
            raise SystemExit(
                "Worker candidate limit must be at least budget_per_round"
            )
        config = replace(config, candidate_limit=args.worker_candidate_limit)
    if args.worker_max_tokens is not None:
        try:
            config = apply_token_budget(config, args.worker_max_tokens)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        os.environ["FITNESS_AGENTS_LLM_MAX_TOKENS"] = str(args.worker_max_tokens)
    if args.disable_batch_control_review or args.disable_batch_diversity_review:
        config = replace(
            config,
            critic=replace(
                config.critic,
                review_controls=(
                    config.critic.review_controls
                    and not args.disable_batch_control_review
                ),
                review_diversity=(
                    config.critic.review_diversity
                    and not args.disable_batch_diversity_review
                ),
            ),
        )
    if args.worker_placeholder_predictor:
        raise SystemExit(
            "Formal hierarchical runs cannot use --placeholder-predictor; "
            "the campaign scores candidates with the configured fitness model"
        )
    summary = run_campaign(config)
    summary["placeholder_predictor"] = False
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run kg_base, kg_base_rag, kg_base_al, then kg_3features_rag across "
            "folds of one GB1 AL96 task with real DeepSeek calls and the configured "
            "Kermut dry validator. Default 4 conditions x 3 folds = 12 jobs; "
            "--max-parallel 4 runs three waves. kg_3features_rag fans out three "
            "child LLM calls; --max-parallel 2 is the rate-limit fallback. "
            "Kermut stays device: cuda:0 inside each worker; pass "
            "--cuda-devices 0,1,2,3 so four concurrent jobs use four cards."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/hierarchical_scientist.deepseek.yaml",
    )
    parser.add_argument("--folds", default="0,1,2", help="all or comma-separated fold indices")
    parser.add_argument(
        "--conditions",
        default=",".join(DEFAULT_CONDITIONS),
        help=(
            "Comma-separated conditions in start order. Default: "
            "kg_base,kg_base_rag,kg_base_al,kg_3features_rag"
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rounds", type=int, help="Override campaign rounds")
    parser.add_argument(
        "--candidate-limit",
        type=int,
        help="Override the config-driven per-round candidate scoring budget",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=FORMAL_DEFAULT_MAX_TOKENS,
        help=(
            "Output token budget for Scientist, Critic, and hierarchical child roles. "
            f"Default {FORMAL_DEFAULT_MAX_TOKENS}; hard cap {FORMAL_MAX_TOKENS}."
        ),
    )
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument(
        "--cuda-devices",
        default="auto",
        help=(
            "GPUs for concurrent Kermut workers. auto discovers visible cards; "
            "0,1,2,3 pins four jobs to four physical GPUs via CUDA_VISIBLE_DEVICES; "
            "none inherits the parent environment (all workers share cuda:0)."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--placeholder-predictor",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable-batch-control-review",
        action="store_true",
        help=(
            "Temporarily remove control-feasibility issues/actions from Batch Critic scope; "
            "candidate acquisition quotas are unchanged"
        ),
    )
    parser.add_argument(
        "--disable-batch-diversity-review",
        action="store_true",
        help=(
            "Temporarily remove diversity metrics/issues/actions from Batch Critic scope"
        ),
    )
    parser.add_argument("--worker-fold", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-condition", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-rounds", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-candidate-limit", type=int, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-max-tokens", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-placeholder-predictor", action="store_true", help=argparse.SUPPRESS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_fold is not None:
        _worker(args)
        return
    root = project_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be at least 1")
    if args.placeholder_predictor:
        raise SystemExit(
            "Formal hierarchical runs cannot use --placeholder-predictor; "
            "omit the flag so the configured fitness model scores candidates"
        )
    try:
        max_tokens = _validate_max_tokens(args.max_tokens)
    except ValueError as error:
        raise SystemExit(f"--max-tokens {error}") from error
    conditions = _parse_conditions(args.conditions)
    try:
        cuda_spec = parse_cuda_devices_arg(args.cuda_devices)
        cuda_devices = resolve_cuda_device_pool(
            cuda_spec,
            max_parallel=args.max_parallel,
            enforce_capacity=not args.dry_run or cuda_spec != "auto",
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not args.dry_run:
        try:
            require_prebuilt_rag_corpus(conditions)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    config = apply_token_budget(load_experiment_config(config_path), max_tokens)
    if args.candidate_limit is not None:
        if args.candidate_limit < config.budget_per_round:
            raise SystemExit("--candidate-limit must be at least budget_per_round")
        config = replace(config, candidate_limit=args.candidate_limit)
    config = validate_formal_fitness_configuration(config)
    expected_rounds = config.rounds if args.rounds is None else args.rounds
    if expected_rounds < 1:
        raise SystemExit("--rounds must be positive")
    if config.task.split_root is None:
        raise SystemExit("Hierarchical fold runner requires a manifest-backed task with split_root")
    manifest_path = config.task.split_root / "manifest.public.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Split manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_folds = int(manifest["n_folds"])
    folds = _parse_folds(args.folds, n_folds)
    if manifest.get("strategy") != config.task.expected_split_strategy:
        raise SystemExit("Split manifest strategy differs from task configuration")
    if manifest.get("protocol_version") != config.task.expected_protocol_version:
        raise SystemExit("Split manifest protocol differs from task configuration")
    seed = config.seed if args.seed is None else args.seed
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or root / "artifacts" / f"hierarchical-scientist-{stamp}"
    jobs = _build_jobs(
        script_path=Path(__file__).resolve(),
        config_path=config_path.resolve(),
        conditions=conditions,
        folds=folds,
        seed=seed,
        expected_rounds=expected_rounds,
        expected_budget=config.budget_per_round,
        expected_candidate_limit=config.candidate_limit,
        output_root=(output_dir / "runs").resolve(),
        python_executable=sys.executable,
        placeholder_predictor=False,
        max_tokens=max_tokens,
        disable_batch_control_review=args.disable_batch_control_review,
        disable_batch_diversity_review=args.disable_batch_diversity_review,
    )
    rag_local = (
        _rag_local_knowledge_template()
        if any(CONDITION_SPECS[item].rag for item in conditions)
        else None
    )
    schedule = {
        "schema_version": "hierarchical-scientist-schedule:v1",
        "config": str(config_path.resolve()),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "strategy": manifest["strategy"],
        "protocol_version": manifest["protocol_version"],
        "folds": folds,
        "conditions": conditions,
        "seed": seed,
        "max_parallel": args.max_parallel,
        "cuda_assignment": cuda_assignment_record(
            policy=args.cuda_devices,
            devices=cuda_devices,
            max_parallel=args.max_parallel,
        ),
        "expected_waves": (len(jobs) + args.max_parallel - 1) // args.max_parallel,
        "expected_rounds": expected_rounds,
        "expected_budget": config.budget_per_round,
        "expected_candidate_pool": config.candidate_limit,
        "placeholder_predictor": False,
        "fitness_predictor": config.model.name,
        "generation_predictor_models": [item.name for item in config.generation.predictor_models],
        "use_fitness_predictors": config.generation.use_fitness_predictors,
        "predictor_roles": {
            "agent_uq_acquisition": "disabled",
            "active_learning_acquisition": "kermut",
            "post_selection_dry_validation": "kermut",
            "oracle_measurement": "approved_batch_only",
        },
        "rag_backend": (
            {
                "embedding": rag_local.retrieval.embedding_api_config.model,
                "reranker": rag_local.retrieval.reranker_api_config.model,
                "corpus_index": str(
                    rag_local.corpus_index_path or rag_local.index_path
                ),
                "per_job_overlay": True,
            }
            if rag_local is not None
            else None
        ),
        "kg_tool_call_budget": {
            item: {
                "required": required_tool_calls(
                    CONDITION_SPECS[item],
                    variant_limit=config.kg_interaction.feature_variant_limit,
                    feature_tool_strategy=(
                        config.kg_interaction.feature_tool_strategy
                        if CONDITION_SPECS[item].channels
                        else "context_only"
                    ),
                ),
                "configured": _interaction_for_spec(
                    config, CONDITION_SPECS[item]
                ).max_tool_calls,
            }
            for item in conditions
        },
        "max_tokens": max_tokens,
        "llm_max_tokens": config.llm.max_tokens,
        "critic_max_tokens": config.critic.max_tokens,
        "hierarchical_role_max_tokens": {
            "child_max_tokens": config.hierarchical_hypothesis.child_max_tokens,
            "child_critic_max_tokens": (
                config.hierarchical_hypothesis.child_critic_max_tokens
            ),
            "main_critic_max_tokens": (
                config.hierarchical_hypothesis.main_critic_max_tokens
            ),
        },
        "batch_review_scope": {
            "controls": (
                config.critic.review_controls
                and not args.disable_batch_control_review
            ),
            "diversity": (
                config.critic.review_diversity
                and not args.disable_batch_diversity_review
            ),
        },
        "jobs": [asdict(job) for job in jobs],
    }
    if args.dry_run:
        print(json.dumps(schedule, indent=2, ensure_ascii=False, default=str))
        return
    load_project_env()
    if not resolve_secret(None, "DEEPSEEK_API_KEY"):
        raise SystemExit("Set DEEPSEEK_API_KEY before running hierarchical scientist folds")
    output_dir.mkdir(parents=True, exist_ok=False)
    log_dir = output_dir / "fold_logs"
    (output_dir / "schedule.json").write_text(
        json.dumps(schedule, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    timeout = args.timeout_seconds if args.timeout_seconds > 0 else None
    model_wants_cuda = str(config.model.device).lower() not in {"cpu", "mps"}
    if (
        model_wants_cuda
        and args.max_parallel > 1
        and cuda_devices is None
        and not args.dry_run
    ):
        print(
            "warning: Kermut is configured for CUDA and --max-parallel "
            f"{args.max_parallel}, but no GPU pool was assigned. All workers "
            "will share physical GPU 0. Pass --cuda-devices 0,1,2,3 or auto.",
            file=sys.stderr,
        )
    results = run_fold_jobs(
        jobs,
        max_parallel=args.max_parallel,
        project_dir=root,
        output_dir=log_dir,
        timeout_seconds=timeout,
        cuda_devices=cuda_devices,
    )
    result_payload = [asdict(result) for result in results]
    (output_dir / "fold_results.json").write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    summaries = [item.summary for item in results if item.summary is not None]
    aggregate = aggregate_runs(summaries, output_dir / "aggregate") if summaries else {}
    integrity = paired_fold_integrity(results, folds=folds, conditions=conditions)
    passed_jobs = sum(item.status == "passed" for item in results)
    failed_jobs = sum(item.status != "passed" for item in results)
    experiment_status = "completed" if failed_jobs == 0 and all(item["passed"] for item in integrity) else "partial"
    report = {
        "overall_passed": experiment_status == "completed",
        "experiment_status": experiment_status,
        "passed_jobs": passed_jobs,
        "failed_jobs": failed_jobs,
        "expected": len(jobs),
        "max_parallel": args.max_parallel,
        "paired_fold_integrity": integrity,
        "aggregate": {key: str(value) for key, value in aggregate.items()},
        "results": result_payload,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
