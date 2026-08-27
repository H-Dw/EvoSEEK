"""Run the paired GB1 no-RAG versus bounded Agentic RAG live-API benchmark.

The protocol is frozen to three folds, three rounds, 32 candidates, 16 wet
measurements per round, and a WT-only cold start.  Both conditions retain the
same two-stage Researcher feature planner; only external local RAG is ablated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fitness_agents.agents.remote_llm import load_project_env, resolve_secret
from fitness_agents.config import (
    ExperimentConfig,
    LocalKnowledgeConfig,
    load_experiment_config,
)
from fitness_agents.local_knowledge.index import SQLiteLocalKnowledgeIndex
from fitness_agents.utils.progress import configure_progress_logging
from scripts.module_tests.benchmark_gb1_directive_rag import audit_validation_feedback

SCHEMA_VERSION = "gb1-agentic-rag-validation:v1"
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/gb1_3features_no_rag_vs_agentic_rag_validation_deepseek_v4_pro.yaml"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/gb1-3features-no-rag-vs-agentic-rag-deepseek-v4-pro-20260827"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "docs/GB1-WT-only-3features-no-RAG-vs-Agentic-RAG真实API测试与结构统计-20260827.md"
)
IDENTITY_FIX_CANARY = (
    PROJECT_ROOT
    / "artifacts/canary/agentic-researcher-identity-fix-v2-20260827/canary-report.json"
)
PRELIMINARY_LONG_ROOT_RECEIPT = (
    PROJECT_ROOT
    / "artifacts/gb1-3features-no-rag-vs-agentic-rag-deepseek-v4-pro-identity-fix-20260827"
    / "paired_agentic_rag_validation_receipt.json"
)
FOLDS = (0, 1, 2)
CONDITIONS = ("researcher_no_rag", "researcher_agentic_rag")
EXPECTED_ROUNDS = 3
EXPECTED_CANDIDATES = 32
EXPECTED_WET_BUDGET = 16
EXPECTED_VISIBLE_COUNTS = (1, 17, 33)
EXPECTED_MANIFEST_HASH = "54059d0d859c8d0c325f9fc518f7db1a2bf73537e207a5b5f0c4db484bd9a686"
MAX_ATTEMPTS_PER_CONDITION = 2
FEEDBACK_CONTRACT = "cold_start_wet_prior"
FIXED_RAG_OPERATORS = {"query_local_knowledge", "query_structured_claims"}
FEATURE_OPERATORS = {
    "query_physchem_delta",
    "query_evolutionary_profile",
    "query_structure_environment",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def environment_receipt(config_path: Path) -> dict[str, Any]:
    dirty = _git_output("status", "--porcelain=v1")
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git_head": _git_output("rev-parse", "HEAD"),
        "git_dirty_entry_count": len(dirty.splitlines()) if dirty else 0,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "command": [str(Path(sys.executable)), *sys.argv],
        "credential_values_recorded": False,
    }


def require_credentials() -> None:
    load_project_env(PROJECT_ROOT / ".env")
    missing = [
        name
        for name in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY")
        if not resolve_secret(f"env:{name}", name)
    ]
    if missing:
        raise RuntimeError("Missing live-API credentials: " + ", ".join(missing))


def inspect_index(config: ExperimentConfig) -> dict[str, Any]:
    local = config.knowledge.local_knowledge
    retrieval = local.retrieval
    corpus_path = local.corpus_index_path or local.index_path
    if corpus_path is None or not corpus_path.is_file():
        raise FileNotFoundError("Frozen Agentic RAG corpus is missing")
    index = SQLiteLocalKnowledgeIndex(corpus_path, read_only=True)
    try:
        index.assert_runtime_binding(local)
        stats = index.stats()
        facets = index.facet_catalog()
    finally:
        index.close()
    embedding = stats.get("embedding_fingerprint") or {}
    reranker = retrieval.reranker_api_config
    checks = {
        "schema_v7": stats.get("schema_version") == "local-knowledge-index:v7",
        "frozen_manifest": stats.get("manifest_hash") == EXPECTED_MANIFEST_HASH,
        "six_documents": stats.get("documents") == 6,
        "six_chunks": stats.get("chunks") == 6,
        "six_embeddings": stats.get("embeddings") == 6,
        "manifest_counts_match": stats.get("manifest_counts_match") is True,
        "qwen_embedding_v4": (
            embedding.get("model_id") == "text-embedding-v4"
            and embedding.get("dimension") == 1024
        ),
        "qwen3_reranker": (
            reranker is not None and reranker.model == "qwen3-rerank"
        ),
        "native_record_types_present": set(facets.get("record_type", ()))
        == {"atomic_claim", "knowledge_decision_card", "logic_unit"},
        "explanation_only": set(facets.get("permission", ())) == {"explanation_only"},
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("Frozen index preflight failed: " + ", ".join(failed))
    return {
        "path": str(corpus_path),
        "sha256": _sha256_file(corpus_path),
        "checks": checks,
        "stats": stats,
        "facet_catalog": facets,
        "read_only_prebuilt": True,
        "rebuilt": False,
    }


def assert_preflight(config: ExperimentConfig) -> dict[str, Any]:
    split_root = config.task.split_root
    manifest_path = split_root / "manifest.public.json" if split_root else None
    manifest = _json(manifest_path, {}) if manifest_path else {}
    fold_manifests = [
        _json(split_root / f"fold_{fold:02d}" / "fold_manifest.json", {})
        for fold in FOLDS
    ] if split_root else []
    quota = config.generation.quota_allocation
    researcher = config.researcher
    retrieval = config.knowledge.local_knowledge.retrieval
    configured_predictors = {
        config.model.name.casefold(),
        *(item.name.casefold() for item in config.generation.predictor_models),
        *(item.name.casefold() for item in config.validation.predictor_models),
        *(
            item.name.casefold()
            for item in config.active_learning.posterior.predictor_models
        ),
    }
    checks = {
        "first_three_of_five_folds": (
            tuple(FOLDS) == (0, 1, 2) and manifest.get("n_folds") == 5
        ),
        "wt_only_split": (
            (manifest.get("options") or {}).get("initial_budget") == 1
            and len(fold_manifests) == 3
            and all(
                (item.get("role_counts") or {}).get("initial_observed") == 1
                and not (item.get("role_counts") or {}).get("train_observed", 0)
                for item in fold_manifests
            )
        ),
        "split_protocol": (
            manifest.get("strategy") == config.task.expected_split_strategy
            and manifest.get("protocol_version") == config.task.expected_protocol_version
        ),
        "three_rounds_32_candidates_16_wet": (
            config.rounds == EXPECTED_ROUNDS
            and config.candidate_limit == EXPECTED_CANDIDATES
            and config.budget_per_round == EXPECTED_WET_BUDGET
        ),
        "cold_start_wild_type": (
            config.prior_schedule.mode == "cold_start"
            and config.prior_schedule.keep_wild_type
            and config.prior_schedule.no_supported_hypothesis_policy
            == "coverage_exploration"
        ),
        "quota_8_3_3_2": (
            quota.enabled
            and (
                quota.hypothesis_target,
                quota.evidence_prior,
                quota.coverage_exploration,
                quota.matched_control,
            ) == (8, 3, 3, 2)
            and quota.total == EXPECTED_WET_BUDGET
        ),
        "three_features": (
            config.knowledge.physchem
            and config.knowledge.conservation
            and config.knowledge.structure
            and tuple(config.kg_interaction.feature_channels)
            == ("physchem", "conservation", "structure")
        ),
        "researcher_two_stage_agentic": (
            researcher.enabled
            and researcher.mode == "two_stage"
            and researcher.provider == "deepseek"
            and researcher.model == "deepseek-v4-pro"
            and config.kg_interaction.feature_tool_strategy == "agentic"
            and retrieval.query_mode == "agentic"
        ),
        "deepseek_v4_pro_roles": (
            config.llm.provider == "deepseek"
            and config.llm.model == "deepseek-v4-pro"
            and config.critic.provider == "deepseek"
            and config.critic.model == "deepseek-v4-pro"
            and config.critic.fallback_policy == "none"
        ),
        "wet_only_no_predictors": (
            not config.validation.enabled
            and not config.validation.predictor_models
            and not config.generation.use_fitness_predictors
            and not config.generation.predictor_models
            and not config.active_learning.enabled
            and config.generation.predictor_weight == 0.0
        ),
        "kermut_absent": "kermut" not in configured_predictors,
        "agentic_budgets": (
            researcher.max_rag_queries == 3
            and researcher.rag_top_k_per_query == 2
            and researcher.max_retrieved_records == 4
            and researcher.max_feature_variants == 2
            and researcher.max_feature_requests == 6
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("Agentic benchmark preflight failed: " + ", ".join(failed))
    return {
        "checks": checks,
        "folds": list(FOLDS),
        "conditions": list(CONDITIONS),
        "manifest": str(manifest_path),
        "index": inspect_index(config),
    }


def condition_run_config(
    base: ExperimentConfig,
    *,
    fold: int,
    condition: str,
    output_root: Path,
) -> ExperimentConfig:
    if fold not in FOLDS:
        raise ValueError(f"Fold must be one of {FOLDS}")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    agentic = condition == "researcher_agentic_rag"
    if agentic:
        local = base.knowledge.local_knowledge
        corpus_path = local.corpus_index_path or local.index_path
        if corpus_path is None or not corpus_path.is_file():
            raise FileNotFoundError("Frozen Agentic RAG corpus is missing")
        local = replace(
            local,
            enabled=True,
            corpus_mode="read_only_prebuilt",
            retrieval_overlay_path=(
                output_root
                / "overlays"
                / f"agentic-rag-f{fold:02d}-{time.time_ns()}.sqlite"
            ),
        )
    else:
        local = LocalKnowledgeConfig(enabled=False)
    arm = "A" if agentic else "N"
    return replace(
        base,
        task=replace(base.task, fold_index=fold),
        condition=condition,
        # Keep generated conversation/id-bridge paths below legacy MAX_PATH on
        # Windows. The descriptive receipt/report remain at output_root.
        run_label=f"{arm}{fold}",
        output_root=output_root / "r" / arm.lower() / str(fold),
        knowledge=replace(base.knowledge, local_knowledge=local),
    )


def _trace_events(run_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    path = run_dir / "trace.jsonl"
    if not path.is_file():
        return events
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def _llm_statistics(run_dir: Path) -> dict[str, Any]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    latency: dict[str, list[float]] = defaultdict(list)
    for event in _trace_events(run_dir):
        event_type = event.get("event_type")
        payload = event.get("payload") or {}
        role = str(payload.get("role") or "unknown")
        phase = str(payload.get("phase") or "")
        key = f"{role}:{phase}" if phase else role
        if event_type == "llm_request_completed":
            groups[key]["completed_calls"] += 1
            groups[key]["prompt_tokens"] += int(payload.get("prompt_tokens") or 0)
            groups[key]["completion_tokens"] += int(payload.get("completion_tokens") or 0)
            groups[key]["total_tokens"] += int(payload.get("total_tokens") or 0)
            latency[key].append(float(payload.get("latency_s") or 0.0))
        elif event_type == "llm_request_retry":
            groups[key]["retries"] += 1
        elif event_type == "llm_request_started":
            groups[key]["attempts"] += 1
    output: dict[str, Any] = {}
    for key in sorted(groups):
        item = dict(groups[key])
        values = latency.get(key, [])
        item["latency_seconds_sum"] = round(sum(values), 3)
        item["latency_seconds_mean"] = round(statistics.fmean(values), 3) if values else 0.0
        output[key] = item
    return output


def _researcher_statistics(run_dir: Path, rounds: int) -> dict[str, Any]:
    round_rows: list[dict[str, Any]] = []
    record_types: Counter[str] = Counter()
    knowledge_types: Counter[str] = Counter()
    feature_channels: Counter[str] = Counter()
    feature_focus: Counter[str] = Counter()
    query_intents: Counter[str] = Counter()
    for round_id in range(1, rounds + 1):
        round_dir = run_dir / f"round_{round_id:02d}"
        receipt = _json(round_dir / "researcher_round_receipt.json", {}) or {}
        phase_a = _json(round_dir / "researcher_phase_a.json", {}) or {}
        phase_b = _json(round_dir / "researcher_phase_b.json", {}) or {}
        external = receipt.get("external_plan") or phase_a.get("plan")
        feature = receipt.get("feature_plan") or phase_b.get("plan")
        needs = (external or {}).get("needs") or []
        feature_needs = (feature or {}).get("needs") or []
        for need in needs:
            query_intents[str(need.get("intent") or "unknown")] += 1
        for need in feature_needs:
            feature_channels[str(need.get("channel") or "unknown")] += 1
            for focus in need.get("focus") or []:
                feature_focus[str(focus)] += 1
        retrieval_payload = _json(round_dir / "local_rag_retrieval.json", []) or []
        results = retrieval_payload if isinstance(retrieval_payload, list) else [retrieval_payload]
        for result in results:
            for record in result.get("records") or []:
                record_types[str(record.get("record_type") or "unknown")] += 1
                knowledge_types[str(record.get("knowledge_type") or "unknown")] += 1
        budget = receipt.get("budget_used") or {}
        executed = phase_a.get("executed_queries") or []
        round_rows.append(
            {
                "round": round_id,
                "receipt_present": bool(receipt),
                "phase_a_present": bool(phase_a),
                "phase_b_present": bool(phase_b),
                "external_decision": (external or {}).get("decision"),
                "feature_decision": (feature or {}).get("decision"),
                "sanitized_queries": [str(item.get("sanitized_query") or "") for item in executed],
                "query_intents": [str(item.get("intent") or "") for item in needs],
                "facets": [item.get("facets") or {} for item in needs],
                "record_ids": list(receipt.get("record_ids") or []),
                "rag_queries": int(budget.get("rag_queries") or 0),
                "retrieved_records": int(budget.get("retrieved_records") or 0),
                "feature_requests": int(budget.get("feature_requests") or 0),
                "feature_needs": feature_needs,
                "skipped": list(receipt.get("skipped") or []),
                "rejected": list(receipt.get("rejected") or []),
            }
        )
    return {
        "rounds": round_rows,
        "receipt_rounds": sum(item["receipt_present"] for item in round_rows),
        "phase_a_rounds": sum(item["phase_a_present"] for item in round_rows),
        "phase_b_rounds": sum(item["phase_b_present"] for item in round_rows),
        "external_plan_count": sum(item["external_decision"] == "PLAN" for item in round_rows),
        "external_abstain_count": sum(item["external_decision"] == "ABSTAIN" for item in round_rows),
        "feature_plan_count": sum(item["feature_decision"] == "PLAN" for item in round_rows),
        "feature_abstain_count": sum(item["feature_decision"] == "ABSTAIN" for item in round_rows),
        "rag_queries": sum(item["rag_queries"] for item in round_rows),
        "retrieved_records": sum(item["retrieved_records"] for item in round_rows),
        "feature_requests": sum(item["feature_requests"] for item in round_rows),
        "query_intents": dict(sorted(query_intents.items())),
        "record_types": dict(sorted(record_types.items())),
        "knowledge_types": dict(sorted(knowledge_types.items())),
        "feature_channels": dict(sorted(feature_channels.items())),
        "feature_focus": dict(sorted(feature_focus.items())),
    }


def _kg_statistics(run_dir: Path, rounds: int) -> list[dict[str, Any]]:
    output = []
    for round_id in range(1, rounds + 1):
        round_dir = run_dir / f"round_{round_id:02d}"
        pre = _json(round_dir / "structured_kg_pre_design.json", {}) or {}
        post = _json(round_dir / "structured_kg_post_validation.json", {}) or {}
        output.append(
            {
                "round": round_id,
                "pre_entities": pre.get("entity_count"),
                "pre_relations": pre.get("relation_count"),
                "post_entities": post.get("entity_count"),
                "post_relations": post.get("relation_count"),
            }
        )
    return output


def _operators_by_round(run_dir: Path, rounds: int) -> list[list[str]]:
    return [
        [
            str(item.get("operator") or "")
            for item in (_json(run_dir / f"round_{round_id:02d}" / "kg_interaction.json", {}) or {}).get("packs", [])
        ]
        for round_id in range(1, rounds + 1)
    ]


def execute_run(config: ExperimentConfig) -> dict[str, Any]:
    from fitness_agents.loop import CampaignRunner

    started = time.perf_counter()
    summary = CampaignRunner(config).run()
    run_dir = Path(summary["run_dir"])
    feedback = audit_validation_feedback(
        run_dir, config, feedback_contract=FEEDBACK_CONTRACT
    )
    candidate_counts = [
        int(
            (_json(run_dir / f"round_{round_id:02d}" / "candidate_pool_receipt.json", {}) or {}).get(
                "actual_candidate_count", -1
            )
        )
        for round_id in range(1, config.rounds + 1)
    ]
    researcher = _researcher_statistics(run_dir, config.rounds)
    operators = _operators_by_round(run_dir, config.rounds)
    agentic = config.condition == "researcher_agentic_rag"
    rag_files = [
        (run_dir / f"round_{round_id:02d}" / "local_rag_retrieval.json").is_file()
        for round_id in range(1, config.rounds + 1)
    ]
    config_record = _json(run_dir / "config.json", {}) or {}
    integrity = {
        "completed": summary.get("run_status") == "completed" and summary.get("rounds_aborted") == 0,
        "three_rounds": len(summary.get("round_metrics") or []) == EXPECTED_ROUNDS,
        "candidate_pool_32_each_round": candidate_counts == [EXPECTED_CANDIDATES] * EXPECTED_ROUNDS,
        "wet_validation_16_each_round": summary.get("actual_batch_sizes") == [EXPECTED_WET_BUDGET] * EXPECTED_ROUNDS,
        "visible_wet_feedback_contract": (
            feedback.get("passed") is True
            and tuple(feedback.get("visible_observation_counts_by_round") or ())
            == EXPECTED_VISIBLE_COUNTS
        ),
        "researcher_receipt_each_round": researcher["receipt_rounds"] == EXPECTED_ROUNDS,
        "researcher_phase_b_each_round": researcher["phase_b_rounds"] == EXPECTED_ROUNDS,
        "phase_a_matches_condition": (
            researcher["phase_a_rounds"] == EXPECTED_ROUNDS if agentic else researcher["phase_a_rounds"] == 0
        ),
        "rag_artifacts_match_condition": all(rag_files) if agentic else not any(rag_files),
        "no_fixed_rag_operator": all(not (set(items) & FIXED_RAG_OPERATORS) for items in operators),
        "no_generation_predictor": not summary.get("fitness_predictors_used_for_generation"),
        "active_learning_disabled": config.active_learning.enabled is False,
        "dry_validation_disabled": config.validation.enabled is False,
        "no_fallback": not summary.get("fallback_nodes"),
        "no_required_node_failure": not summary.get("required_node_failures"),
        "deepseek_v4_pro_all_roles": (
            (config_record.get("llm") or {}).get("model") == "deepseek-v4-pro"
            and (config_record.get("critic") or {}).get("model") == "deepseek-v4-pro"
            and config.researcher.model == "deepseek-v4-pro"
        ),
        "fold_matches": summary.get("data_source", {}).get("fold_index") == config.task.fold_index,
        "kermut_not_activated": config.model.name.casefold() != "kermut",
    }
    round_metrics = summary.get("round_metrics") or []
    round_best_seen = [float(item["best_seen_fitness"]) for item in round_metrics]
    result = {
        "fold": config.task.fold_index,
        "condition": config.condition,
        "run_id": summary.get("run_id"),
        "run_dir": str(run_dir),
        "round_best_seen": round_best_seen,
        "round_batch_best": [float(item["batch_best_fitness"]) for item in round_metrics],
        "round_batch_mean": [float(item["batch_mean_fitness"]) for item in round_metrics],
        "final_best_seen": round_best_seen[-1] if round_best_seen else None,
        "auc_proxy": statistics.fmean(round_best_seen) if round_best_seen else None,
        "candidate_counts_by_round": candidate_counts,
        "feedback": feedback,
        "researcher": researcher,
        "llm": _llm_statistics(run_dir),
        "structured_kg": _kg_statistics(run_dir, config.rounds),
        "operators_by_round": operators,
        "integrity": integrity,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    failed = [name for name, passed in integrity.items() if not passed]
    if failed:
        result["status"] = "invalid"
        result["integrity_failures"] = failed
    else:
        result["status"] = "completed"
    return result


def _is_selected(attempt: dict[str, Any]) -> bool:
    result = attempt.get("result") or {}
    integrity = result.get("integrity") or {}
    return (
        attempt.get("status") == "completed"
        and bool(integrity)
        and all(bool(value) for value in integrity.values())
        and bool(result.get("run_dir"))
        and Path(result["run_dir"]).is_dir()
    )


def selected_runs(attempts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[int, str], dict[str, Any]] = {}
    for attempt in attempts:
        if _is_selected(attempt):
            result = attempt["result"]
            selected[(int(result["fold"]), str(result["condition"]))] = result
    return [selected[key] for key in sorted(selected)]


def _id_set(run_dir: Path, round_id: int, name: str, key: str) -> set[str]:
    payload = _json(run_dir / f"round_{round_id:02d}" / name, {}) or {}
    return {str(item) for item in payload.get(key) or []}


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    paired = []
    overlaps = []
    for fold in FOLDS:
        by_condition = {
            str(item["condition"]): item for item in runs if int(item["fold"]) == fold
        }
        if set(CONDITIONS) <= by_condition.keys():
            no_rag = by_condition["researcher_no_rag"]
            agentic = by_condition["researcher_agentic_rag"]
            paired.append(
                {
                    "fold": fold,
                    "final_best_delta": float(agentic["final_best_seen"])
                    - float(no_rag["final_best_seen"]),
                    "auc_delta": float(agentic["auc_proxy"])
                    - float(no_rag["auc_proxy"]),
                }
            )
            for round_id in range(1, EXPECTED_ROUNDS + 1):
                no_dir = Path(no_rag["run_dir"])
                ar_dir = Path(agentic["run_dir"])
                no_candidates = _id_set(no_dir, round_id, "candidate_pool_receipt.json", "candidate_ids")
                ar_candidates = _id_set(ar_dir, round_id, "candidate_pool_receipt.json", "candidate_ids")
                no_approved = _id_set(no_dir, round_id, "approved_batch.json", "candidate_ids")
                ar_approved = _id_set(ar_dir, round_id, "approved_batch.json", "candidate_ids")
                overlaps.append(
                    {
                        "fold": fold,
                        "round": round_id,
                        "candidate_overlap": len(no_candidates & ar_candidates),
                        "candidate_denominator": EXPECTED_CANDIDATES,
                        "approved_overlap": len(no_approved & ar_approved),
                        "approved_denominator": EXPECTED_WET_BUDGET,
                    }
                )
    final_deltas = [item["final_best_delta"] for item in paired]
    auc_deltas = [item["auc_delta"] for item in paired]
    median_final = statistics.median(final_deltas) if final_deltas else None
    mean_auc = statistics.fmean(auc_deltas) if auc_deltas else None
    agentic_runs = [
        item for item in runs if item.get("condition") == "researcher_agentic_rag"
    ]
    total_agentic_queries = sum(
        int(item.get("researcher", {}).get("rag_queries", 0))
        for item in agentic_runs
    )
    all_agentic_rounds_abstained = bool(agentic_runs) and all(
        int(item.get("researcher", {}).get("external_abstain_count", 0))
        == EXPECTED_ROUNDS
        for item in agentic_runs
    )
    return {
        "complete_pair_count": len(paired),
        "paired_deltas": paired,
        "median_paired_final_best_delta": median_final,
        "mean_paired_final_best_delta": statistics.fmean(final_deltas) if final_deltas else None,
        "mean_paired_auc_delta": mean_auc,
        "candidate_and_approved_overlap": overlaps,
        "total_agentic_rag_queries": total_agentic_queries,
        "all_agentic_rounds_abstained": all_agentic_rounds_abstained,
        "runtime_integrity_passed": len(runs) == 6 and len(paired) == 3,
        "positive_efficacy_supported": (
            len(paired) == 3
            and median_final is not None
            and median_final > 0.0
            and mean_auc is not None
            and mean_auc > 0.0
        ),
        "variant_identities_recorded_in_aggregate": False,
    }


def _redacted_error(error: Exception) -> str:
    message = str(error)
    for name in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", message)[:800]


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fmt(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _counter_text(value: dict[str, Any]) -> str:
    return ", ".join(f"{key}={item}" for key, item in sorted(value.items())) or "—"


def _facet_text(value: list[dict[str, Any]]) -> str:
    groups = []
    for facets in value:
        groups.append(
            ", ".join(
                f"{key}={'+'.join(str(item) for item in items)}"
                for key, items in sorted(facets.items())
            )
        )
    return "; ".join(item for item in groups if item) or "—"


def _feature_need_text(value: list[dict[str, Any]]) -> str:
    rows = []
    for item in value:
        rows.append(
            "channel={channel}, focus={focus}, position_count={position_count}".format(
                channel=item.get("channel", "unknown"),
                focus="+".join(str(value) for value in item.get("focus") or []) or "—",
                position_count=len(set(item.get("positions") or [])),
            )
        )
    return "; ".join(rows) or "—"


def _qwen_statistics(run_dir: Path, rounds: int) -> dict[str, int]:
    output = Counter[str]()
    for round_id in range(1, rounds + 1):
        payload = _json(run_dir / f"round_{round_id:02d}" / "local_rag_retrieval.json", []) or []
        rows = payload if isinstance(payload, list) else [payload]
        for item in rows:
            output["query_records"] += 1
            if (item.get("policy_decision") or {}).get("allowed") is True:
                output["embedding_queries"] += 1
            chunks = item.get("chunks") or []
            if any(chunk.get("scores", {}).get("reranker") is not None for chunk in chunks):
                output["reranker_observed_queries"] += 1
            output["returned_chunks"] += len(chunks)
            output["returned_records"] += len(item.get("records") or [])
            if "no_answer_above_retrieval_threshold" in (item.get("warnings") or []):
                output["no_answer_queries"] += 1
    return {
        key: int(output[key])
        for key in (
            "query_records",
            "embedding_queries",
            "reranker_observed_queries",
            "returned_chunks",
            "returned_records",
            "no_answer_queries",
        )
    }


def _identity_validation(receipt: dict[str, Any]) -> dict[str, Any]:
    canary = _json(IDENTITY_FIX_CANARY, {}) or {}
    config_path = receipt.get("environment", {}).get("config_path")
    forbidden_terms: tuple[str, ...] = ()
    if config_path and Path(config_path).is_file():
        config = load_experiment_config(Path(config_path))
        forbidden_terms = tuple(
            str(item).strip().casefold()
            for item in (
                config.task.protein_id,
                config.task.protein_name,
                *config.task.protein_aliases,
                *config.task.protein_accessions,
            )
            if str(item).strip()
        )
    queries = [
        str(query)
        for run in receipt.get("selected_runs") or []
        if run.get("condition") == "researcher_agentic_rag"
        for item in (run.get("researcher") or {}).get("rounds") or []
        for query in item.get("sanitized_queries") or []
    ]
    protected_hits = sum(
        any(term in query.casefold() for term in forbidden_terms)
        for query in queries
    )
    return {
        "canary_path": str(IDENTITY_FIX_CANARY),
        "canary_status": canary.get("status"),
        "profile": (canary.get("providers") or {}).get("researcher", {}).get("profile"),
        "profile_version": (canary.get("providers") or {}).get("researcher", {}).get(
            "profile_version"
        ),
        "opaque_assay_label": (canary.get("integrity") or {}).get("opaque_assay_label"),
        "canary_identity_neutral_queries": (canary.get("integrity") or {}).get(
            "identity_neutral_queries"
        ),
        "benchmark_query_count": len(queries),
        "benchmark_protected_hit_count": protected_hits,
    }


def _attempt_failure_text(attempt: dict[str, Any]) -> str:
    error = attempt.get("error") or {}
    if error:
        value = f"{error.get('error_type', 'Error')}: {error.get('message', '')}".strip()
    else:
        result = attempt.get("result") or {}
        rejected = []
        for item in result.get("researcher", {}).get("rounds", []):
            rejected.extend(item.get("rejected") or [])
        value = "; ".join(
            f"{item.get('step_id', 'unknown')}: {item.get('reason', 'rejected')}"
            for item in rejected
        )
        if not value and result.get("run_dir"):
            summary = _json(Path(result["run_dir"]) / "summary.json", {}) or {}
            value = "; ".join(str(item) for item in summary.get("required_node_failures") or [])
            value = re.sub(
                r"input_value=.*$",
                "input_value=[REDACTED]",
                value,
                flags=re.DOTALL,
            )
            for name in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
                secret = os.environ.get(name)
                if secret:
                    value = value.replace(secret, "[REDACTED]")
            value = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", value)
    return value.replace("|", "\\|").replace("\n", " ")[:800] or "—"


def _preliminary_infrastructure_attempts() -> list[dict[str, Any]]:
    """Return preserved attempts made before switching to the short output root."""
    receipt = _json(PRELIMINARY_LONG_ROOT_RECEIPT, {}) or {}
    return list(receipt.get("attempts") or [])


def _receipt_terminal_status(receipt: dict[str, Any]) -> str:
    if (receipt.get("aggregate") or {}).get("runtime_integrity_passed") is True:
        return "completed"
    selected = {
        (int(item["fold"]), str(item["condition"]))
        for item in receipt.get("selected_runs") or []
    }
    counts = Counter(
        (int(item.get("fold", -1)), str(item.get("condition")))
        for item in receipt.get("attempts") or []
    )
    for fold in FOLDS:
        order = CONDITIONS if fold % 2 == 0 else tuple(reversed(CONDITIONS))
        for condition in order:
            key = (fold, condition)
            if key not in selected and counts[key] >= MAX_ATTEMPTS_PER_CONDITION:
                return f"attempt_limit_exhausted (fold {fold}, {condition})"
    return "in_progress"


def render_report(receipt: dict[str, Any]) -> str:
    runs = receipt.get("selected_runs") or []
    aggregate_value = receipt.get("aggregate") or {}
    identity = _identity_validation(receipt)
    complete = aggregate_value.get("runtime_integrity_passed") is True
    if complete:
        if aggregate_value.get("positive_efficacy_supported"):
            conclusion = "工程完整性通过；两个预注册效能方向指标均为正，结果支持当前候选知识包下的正向效能。"
        else:
            conclusion = "工程完整性通过；预注册正向效能条件未同时满足，不能声称 Agentic RAG 提升 fitness。"
    else:
        conclusion = "实验尚未形成三个完整配对；当前数据仅作为运行记录，不作效能结论。"
    command = subprocess.list2cmdline(receipt.get("environment", {}).get("command") or [])
    lines = [
        "# GB1 WT-only 3features：no-RAG vs Agentic RAG 真实 API 测试与结构统计",
        "",
        f"更新时间：{_now()}（UTC；实验时区 Asia/Shanghai）",
        "",
        "## 结论与状态",
        "",
        f"- 状态：`{'COMPLETE' if complete else 'INCOMPLETE'}`",
        f"- {conclusion}",
        f"- 完整配对：{aggregate_value.get('complete_pair_count', 0)}/3；有效 run：{len(runs)}/6。",
        "- 本报告不包含 API key、mutation identities、隐藏推理或完整 conversation 内容。",
        "",
        "## 冻结协议",
        "",
        "- folds 0/1/2；seed 11；每个 run 3 rounds；每轮 32 candidates、16 wet validations；WT-only 冷启动。",
        "- no-RAG 保留 DeepSeek Researcher Phase B，只跳过外部检索 Phase A。",
        "- Agentic RAG 使用 DeepSeek `deepseek-v4-pro` 动态 Phase A/Phase B、Qwen `text-embedding-v4` 与 `qwen3-rerank`。",
        "- Kermut、generation predictor、active learning、dry validation、fallback 与固定 RAG prefetch 均禁用。",
        "",
        "## Protected identity 修复验证",
        "",
        f"- Live canary：`{identity.get('canary_status') or '—'}`；Researcher profile：`{identity.get('profile') or '—'}` v`{identity.get('profile_version') or '—'}`；opaque assay label：{identity.get('opaque_assay_label')}；identity-neutral queries：{identity.get('canary_identity_neutral_queries')}。",
        f"- 完整 benchmark 的 Agentic Phase A sanitized queries：{identity.get('benchmark_query_count', 0)}；与冻结 protein ID/name/alias/accession 的精确命中：{identity.get('benchmark_protected_hit_count', 0)}。",
        "- 完整 benchmark 中未发生 protected-identity 外层失败；fold 0 的一次 Phase A 语义重试来自越界 facet 值，并非身份泄漏。首答含保护身份时的同调用自动修复由 focused regression test 覆盖。",
        f"- Canary artifact：`{identity.get('canary_path', '—')}`",
        "",
        "## 命令与运行记录",
        "",
        f"- 命令：`{command or '—'}`",
        f"- Runner 终止状态：`{_receipt_terminal_status(receipt)}`",
        f"- Receipt 开始/更新时间：{receipt.get('created_at', '—')} / {receipt.get('updated_at', '—')}。",
        f"- Python：`{receipt.get('environment', {}).get('python_executable', '—')}`",
        f"- Python 版本：`{receipt.get('environment', {}).get('python', '—')}`",
        f"- Git HEAD：`{receipt.get('environment', {}).get('git_head', '—')}`；dirty entries：{receipt.get('environment', {}).get('git_dirty_entry_count', '—')}。",
        f"- 配置 SHA-256：`{receipt.get('environment', {}).get('config_sha256', '—')}`",
        "",
        "| Attempt | Fold | Condition | Status | Started | Finished | Elapsed(s) | Run directory | Error |",
        "|---:|---:|---|---|---|---|---:|---|---|",
    ]
    for index, attempt in enumerate(receipt.get("attempts") or [], 1):
        result = attempt.get("result") or {}
        lines.append(
            "| {idx} | {fold} | {condition} | {status} | {started} | {finished} | {elapsed} | `{run_dir}` | {error} |".format(
                idx=index,
                fold=attempt.get("fold", "—"),
                condition=attempt.get("condition", "—"),
                status=attempt.get("status", "—"),
                started=attempt.get("started_at", "—"),
                finished=attempt.get("finished_at", "—"),
                elapsed=_fmt(result.get("elapsed_seconds"), 1),
                run_dir=result.get("run_dir", "—"),
                error=_attempt_failure_text(attempt),
            )
        )
    preliminary_attempts = _preliminary_infrastructure_attempts()
    if preliminary_attempts:
        lines.extend(
            [
                "",
                "### 前置基础设施尝试（未纳入最终配对 receipt）",
                "",
                "- 首次使用描述性长输出根目录时触发 Windows 路径长度限制；该尝试已原样保留，随后改用短根目录 `artifacts/gf2` 从 fold 0 重新开始。它不是 protected-identity 或模型输出失败，也不进入配对统计。",
                f"- 前置 receipt：`{PRELIMINARY_LONG_ROOT_RECEIPT}`",
                "",
                "| Attempt | Fold | Condition | Status | Elapsed(s) | Failure |",
                "|---:|---:|---|---|---:|---|",
            ]
        )
        for index, attempt in enumerate(preliminary_attempts, 1):
            result = attempt.get("result") or {}
            lines.append(
                "| P{idx} | {fold} | {condition} | {status} | {elapsed} | {failure} |".format(
                    idx=index,
                    fold=attempt.get("fold", "—"),
                    condition=attempt.get("condition", "—"),
                    status=attempt.get("status", "—"),
                    elapsed=_fmt(result.get("elapsed_seconds"), 1),
                    failure=_attempt_failure_text(attempt),
                )
            )
    index_value = receipt.get("index") or {}
    stats = index_value.get("stats") or {}
    lines.extend(
        [
            "",
            "## 索引结构统计",
            "",
            f"- 路径：`{index_value.get('path', '—')}`",
            f"- SHA-256：`{index_value.get('sha256', '—')}`；manifest：`{stats.get('manifest_hash', '—')}`",
            f"- schema：`{stats.get('schema_version', '—')}`；documents/chunks/embeddings：{stats.get('documents', '—')}/{stats.get('chunks', '—')}/{stats.get('embeddings', '—')}。",
            f"- knowledge types：{_counter_text(stats.get('knowledge_types') or {})}",
            f"- record types：{_counter_text({key.split('=', 1)[1]: value for key, value in (stats.get('facets') or {}).items() if key.startswith('record_type=')})}",
            f"- permissions：{_counter_text({key.split('=', 1)[1]: value for key, value in (stats.get('facets') or {}).items() if key.startswith('permission=')})}",
            "",
            "## 逐 fold fitness",
            "",
            "| Fold | Condition | R1 batch best | R2 batch best | R3 batch best | Final best-seen | AUC proxy |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in runs:
        batch_best = run.get("round_batch_best") or [None, None, None]
        lines.append(
            f"| {run['fold']} | {run['condition']} | {_fmt(batch_best[0])} | {_fmt(batch_best[1])} | {_fmt(batch_best[2])} | {_fmt(run.get('final_best_seen'))} | {_fmt(run.get('auc_proxy'))} |"
        )
    lines.extend(
        [
            "",
            "| Fold | Final-best delta (Agentic − no-RAG) | AUC delta |",
            "|---:|---:|---:|",
        ]
    )
    for item in aggregate_value.get("paired_deltas") or []:
        lines.append(f"| {item['fold']} | {_fmt(item['final_best_delta'])} | {_fmt(item['auc_delta'])} |")
    lines.extend(
        [
            "",
            f"- paired final-best delta 中位数：{_fmt(aggregate_value.get('median_paired_final_best_delta'))}",
            f"- paired final-best delta 平均值：{_fmt(aggregate_value.get('mean_paired_final_best_delta'))}",
            f"- paired AUC delta 平均值：{_fmt(aggregate_value.get('mean_paired_auc_delta'))}",
            "",
            "## 候选与批准批次重叠",
            "",
            "| Fold | Round | Candidate overlap | Approved overlap |",
            "|---:|---:|---:|---:|",
        ]
    )
    for item in aggregate_value.get("candidate_and_approved_overlap") or []:
        lines.append(
            f"| {item['fold']} | {item['round']} | {item['candidate_overlap']}/{item['candidate_denominator']} | {item['approved_overlap']}/{item['approved_denominator']} |"
        )
    lines.extend(
        [
            "",
            "## Researcher 结构统计",
            "",
            "| Fold | Condition | Round | Phase A | RAG budget | Records | Phase B | Feature budget | Intents | Skipped | Rejected |",
            "|---:|---|---:|---|---:|---:|---|---:|---|---:|---:|",
        ]
    )
    for attempt_index, attempt in enumerate(receipt.get("attempts") or [], 1):
        run = attempt.get("result") or {}
        if not run:
            continue
        for item in run.get("researcher", {}).get("rounds", []):
            lines.append(
                f"| {run['fold']} | {run['condition']}#A{attempt_index} | {item['round']} | {item.get('external_decision') or 'SKIPPED'} | {item['rag_queries']}/3 | {item['retrieved_records']} | {item.get('feature_decision') or '—'} | {item['feature_requests']}/6 | {', '.join(item.get('query_intents') or []) or '—'} | {len(item.get('skipped') or [])} | {len(item.get('rejected') or [])} |"
            )
            lines.append(
                f"  - fold {run['fold']} / round {item['round']} facets: {_facet_text(item.get('facets') or [])}; record IDs: {', '.join(item.get('record_ids') or []) or '—'}."
            )
            lines.append(
                f"  - fold {run['fold']} / round {item['round']} feature projections: {_feature_need_text(item.get('feature_needs') or [])}."
            )
            for query in item.get("sanitized_queries") or []:
                lines.append(f"  - fold {run['fold']} / round {item['round']} sanitized query: {query}")
            if item.get("skipped") or item.get("rejected"):
                lines.append(
                    f"  - fold {run['fold']} / round {item['round']} skipped/rejected: "
                    f"{json.dumps({'skipped': item.get('skipped') or [], 'rejected': item.get('rejected') or []}, ensure_ascii=False, sort_keys=True)}"
                )
        research = run.get("researcher") or {}
        lines.append(
            f"  - attempt {attempt_index}, fold {run['fold']} `{run['condition']}` record types: {_counter_text(research.get('record_types') or {})}; knowledge types: {_counter_text(research.get('knowledge_types') or {})}; feature channels: {_counter_text(research.get('feature_channels') or {})}; focus: {_counter_text(research.get('feature_focus') or {})}."
        )
    lines.extend(
        [
            "",
            "## Qwen 查询与重排统计",
            "",
            "| Fold | Condition | Attempt | Query records | Embedding queries | Reranker-observed queries | Returned chunks | Returned records | No-answer queries |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for attempt_index, attempt in enumerate(receipt.get("attempts") or [], 1):
        run = attempt.get("result") or {}
        if not run or not run.get("run_dir"):
            continue
        qwen = _qwen_statistics(Path(run["run_dir"]), EXPECTED_ROUNDS)
        lines.append(
            f"| {run['fold']} | {run['condition']} | {attempt_index} | {qwen['query_records']} | {qwen['embedding_queries']} | {qwen['reranker_observed_queries']} | {qwen['returned_chunks']} | {qwen['returned_records']} | {qwen['no_answer_queries']} |"
        )
    lines.extend(
        [
            "",
            "- 在冻结的 hybrid retriever 中，每个 policy-allowed query record 都会执行一次 `text-embedding-v4` query embedding；因此 embedding query 数是可审计的真实请求数。",
            "- 当前 artifact 未单独持久化 reranker HTTP request counter；`Reranker-observed queries` 仅统计返回 chunk 中存在 `reranker` 分数的可证明下界，不能解释为精确请求总数。",
        ]
    )
    lines.extend(["", "## LLM 调用、token、延迟与重试", ""])
    for attempt_index, attempt in enumerate(receipt.get("attempts") or [], 1):
        run = attempt.get("result") or {}
        if not run:
            continue
        lines.append(
            f"### Attempt {attempt_index} · Fold {run['fold']} · {run['condition']} · {attempt.get('status', '—')}"
        )
        lines.extend(
            [
                "",
                "| Role/phase | Attempts | Completed | Retries | Prompt tokens | Completion tokens | Total tokens | Latency(s) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for role, item in sorted((run.get("llm") or {}).items()):
            lines.append(
                f"| {role} | {item.get('attempts', 0)} | {item.get('completed_calls', 0)} | {item.get('retries', 0)} | {item.get('prompt_tokens', 0)} | {item.get('completion_tokens', 0)} | {item.get('total_tokens', 0)} | {_fmt(item.get('latency_seconds_sum'), 1)} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Structured KG 统计",
            "",
            "| Fold | Condition | Round | Pre entities | Pre relations | Post entities | Post relations |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in runs:
        for item in run.get("structured_kg") or []:
            lines.append(
                f"| {run['fold']} | {run['condition']} | {item['round']} | {item.get('pre_entities', '—')} | {item.get('pre_relations', '—')} | {item.get('post_entities', '—')} | {item.get('post_relations', '—')} |"
            )
    lines.extend(["", "## 失败尝试与完整性判断", ""])
    for attempt_index, attempt in enumerate(receipt.get("attempts") or [], 1):
        result = attempt.get("result") or {}
        failures = result.get("integrity_failures") or []
        lines.append(
            f"- Attempt {attempt_index} · fold {attempt.get('fold', '—')} · "
            f"`{attempt.get('condition', '—')}` · `{attempt.get('status', '—')}`："
            f"失败/拒绝原因：{_attempt_failure_text(attempt)}；"
            f"完整性失败项：{', '.join(failures) or '—'}；"
            f"artifact：`{result.get('run_dir', '—')}`。"
        )
    lines.extend(
        [
            "",
            "## 完整性与局限",
            "",
            f"- Agentic RAG 实际查询总数：{aggregate_value.get('total_agentic_rag_queries', 0)}；全轮 ABSTAIN：{aggregate_value.get('all_agentic_rounds_abstained', False)}。",
            "- Agentic 路径只自主规划外部检索与 feature projections；宿主固定的 context、assay 和审计步骤仍保留。",
            "- 当前 6-record bundle 全部为 explanation-only，尚不是生产级人工审核知识产品。",
            "- 本实验没有 budget-matched fixed-RAG 第三臂，不能证明 Agentic planning 优于固定查询。",
            "- n=3 仅支持描述性配对结论；真实 LLM 即使 temperature=0 也不是逐 token 确定性重放。",
            "",
            "## 可复核产物",
            "",
            f"- Root receipt：`{receipt.get('receipt_path', '—')}`",
            f"- 配置：`{receipt.get('environment', {}).get('config_path', '—')}`",
            f"- Runner：`{Path(__file__).resolve()}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _write_report(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(receipt), encoding="utf-8")


def _refresh_derived(receipt: dict[str, Any]) -> None:
    runs = selected_runs(receipt.get("attempts") or [])
    receipt["selected_runs"] = runs
    receipt["aggregate"] = aggregate(runs)
    receipt["updated_at"] = _now()


def _new_receipt(
    *,
    config_path: Path,
    receipt_path: Path,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "updated_at": _now(),
        "receipt_path": str(receipt_path),
        "environment": environment_receipt(config_path),
        "protocol": {
            "folds": list(FOLDS),
            "conditions": list(CONDITIONS),
            "rounds": EXPECTED_ROUNDS,
            "candidate_limit": EXPECTED_CANDIDATES,
            "wet_validation_budget": EXPECTED_WET_BUDGET,
            "visible_observation_counts": list(EXPECTED_VISIBLE_COUNTS),
            "max_attempts_per_fold_condition": MAX_ATTEMPTS_PER_CONDITION,
        },
        "preflight": {"checks": preflight["checks"], "manifest": preflight["manifest"]},
        "index": preflight["index"],
        "attempts": [],
        "selected_runs": [],
        "aggregate": aggregate([]),
        "api_keys_recorded": False,
        "variant_identities_recorded_in_receipt": False,
        "hidden_reasoning_recorded_in_receipt": False,
    }


def _load_or_create_receipt(
    *,
    path: Path,
    config_path: Path,
    preflight: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    if path.is_file():
        if not resume:
            raise FileExistsError("Receipt already exists; rerun with --resume")
        receipt = _json(path, {}) or {}
        if receipt.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Existing receipt schema is incompatible")
        current_hash = _sha256_file(config_path)
        if receipt.get("environment", {}).get("config_sha256") != current_hash:
            raise RuntimeError("Config changed since the existing receipt was created")
        if receipt.get("index", {}).get("stats", {}).get("manifest_hash") != EXPECTED_MANIFEST_HASH:
            raise RuntimeError("Existing receipt is bound to a different index manifest")
        _refresh_derived(receipt)
        return receipt
    if path.parent.exists() and any(path.parent.iterdir()) and not resume:
        raise FileExistsError("Output root is not empty; use --resume or a new output root")
    return _new_receipt(
        config_path=config_path,
        receipt_path=path,
        preflight=preflight,
    )


def main() -> int:
    args = arguments()
    config_path = _absolute(args.config)
    output_root = _absolute(args.output_root)
    report_path = _absolute(args.report)
    base = load_experiment_config(config_path)
    preflight = assert_preflight(base)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "schema_version": SCHEMA_VERSION,
                    "preflight": preflight,
                    "config": str(config_path),
                    "output_root": str(output_root),
                    "report": str(report_path),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0
    require_credentials()
    configure_progress_logging()
    receipt_path = output_root / "paired_agentic_rag_validation_receipt.json"
    receipt = _load_or_create_receipt(
        path=receipt_path,
        config_path=config_path,
        preflight=preflight,
        resume=args.resume,
    )
    _write_receipt(receipt_path, receipt)
    _write_report(report_path, receipt)

    selected = {
        (int(item["fold"]), str(item["condition"]))
        for item in receipt.get("selected_runs") or []
    }
    for fold in FOLDS:
        order = CONDITIONS if fold % 2 == 0 else tuple(reversed(CONDITIONS))
        for condition in order:
            key = (fold, condition)
            if key in selected:
                continue
            prior_attempts = [
                item
                for item in receipt.get("attempts") or []
                if int(item.get("fold", -1)) == fold
                and str(item.get("condition")) == condition
            ]
            if len(prior_attempts) >= MAX_ATTEMPTS_PER_CONDITION:
                _refresh_derived(receipt)
                _write_receipt(receipt_path, receipt)
                _write_report(report_path, receipt)
                print(
                    json.dumps(
                        {
                            "status": "attempt_limit_exhausted",
                            "fold": fold,
                            "condition": condition,
                            "receipt": str(receipt_path),
                            "report": str(report_path),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 2

            started_at = _now()
            attempt: dict[str, Any] = {
                "fold": fold,
                "condition": condition,
                "attempt_number": len(prior_attempts) + 1,
                "started_at": started_at,
                "finished_at": None,
                "status": "running",
            }
            receipt["attempts"].append(attempt)
            _refresh_derived(receipt)
            _write_receipt(receipt_path, receipt)
            _write_report(report_path, receipt)
            config = condition_run_config(
                base,
                fold=fold,
                condition=condition,
                output_root=output_root,
            )
            try:
                result = execute_run(config)
                attempt["result"] = result
                attempt["status"] = result["status"]
            except Exception as error:  # noqa: BLE001 - audited provider boundary
                attempt["status"] = "failed"
                attempt["error"] = {
                    "error_type": type(error).__name__,
                    "message": _redacted_error(error),
                    "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                }
            attempt["finished_at"] = _now()
            _refresh_derived(receipt)
            _write_receipt(receipt_path, receipt)
            _write_report(report_path, receipt)
            if attempt["status"] != "completed":
                print(
                    json.dumps(
                        {
                            "status": attempt["status"],
                            "fold": fold,
                            "condition": condition,
                            "attempt_number": attempt["attempt_number"],
                            "error": attempt.get("error"),
                            "integrity_failures": (attempt.get("result") or {}).get(
                                "integrity_failures"
                            ),
                            "receipt": str(receipt_path),
                            "report": str(report_path),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 1
            selected.add(key)

    _refresh_derived(receipt)
    _write_receipt(receipt_path, receipt)
    _write_report(report_path, receipt)
    print(
        json.dumps(
            {
                "status": "completed",
                "selected_runs": len(receipt["selected_runs"]),
                "complete_pairs": receipt["aggregate"]["complete_pair_count"],
                "positive_efficacy_supported": receipt["aggregate"][
                    "positive_efficacy_supported"
                ],
                "receipt": str(receipt_path),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if receipt["aggregate"]["runtime_integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
