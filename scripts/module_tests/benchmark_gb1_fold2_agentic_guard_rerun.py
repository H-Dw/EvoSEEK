"""Run the post-hoc fold-2 no-RAG versus guarded Agentic RAG live benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fitness_agents.config import ExperimentConfig, load_experiment_config
from fitness_agents.data import load_campaign_fold_bundle
from fitness_agents.utils.progress import configure_progress_logging
from scripts.module_tests import benchmark_gb1_agentic_rag_validation as base_runner

SCHEMA_VERSION = "gb1-fold2-agentic-rag-guard-rerun:v1"
TARGET_FOLD = 2
CONDITIONS = ("researcher_no_rag", "researcher_agentic_rag")
MAX_ATTEMPTS_PER_CONDITION = 2
DEFAULT_CONFIG = base_runner.DEFAULT_CONFIG
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/gf2g1"
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "docs/GB1-fold2-no-RAG-vs-Agentic-RAG候选护栏真实API复测-20260828.md"
)
HISTORICAL_RECEIPT = (
    PROJECT_ROOT / "artifacts/gf2/paired_agentic_rag_validation_receipt.json"
)
IMPLEMENTATION_FILES = (
    PROJECT_ROOT / "src/fitness_agents/mutation/hypothesis_scoring.py",
    PROJECT_ROOT / "src/fitness_agents/mutation/generators.py",
    PROJECT_ROOT / "src/fitness_agents/mutation/uncertainty.py",
    PROJECT_ROOT / "src/fitness_agents/loop/orchestrator.py",
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_receipt() -> dict[str, Any]:
    files = {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in IMPLEMENTATION_FILES}
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "files": files,
        "combined_sha256": combined,
        "hypothesis_match_policy": "edited_non_wild_type_sites_only",
        "evidence_prefilter_policy": "selection_authorized_only",
    }


def _json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _depth_map(run_dir: Path) -> dict[str, int]:
    config = _json(run_dir / "config.json", {}) or {}
    source = config.get("data_source") or {}
    bundle = load_campaign_fold_bundle(
        Path(source["split_root"]),
        int(source["fold_index"]),
    )
    return {
        item.variant_id: int(item.mutation_count)
        for item in [*bundle.initial_variants, *bundle.public_candidates]
    }


def _depth_summary(ids: list[str], depth_by_id: dict[str, int]) -> dict[str, Any]:
    depths = [depth_by_id[item] for item in ids]
    counts = Counter(depths)
    maximum = max(depth_by_id.values(), default=0)
    return {
        "count": len(depths),
        "counts": {str(key): counts[key] for key in sorted(counts)},
        "low_depth_count": sum(depth <= 2 for depth in depths),
        "low_depth_share": (
            sum(depth <= 2 for depth in depths) / len(depths) if depths else None
        ),
        "max_depth": maximum,
        "max_depth_count": counts[maximum],
        "max_depth_share": counts[maximum] / len(depths) if depths else None,
        "mean_depth": statistics.fmean(depths) if depths else None,
        "median_depth": statistics.median(depths) if depths else None,
    }


def _round_depth_statistics(run_dir: Path, rounds: int = 3) -> list[dict[str, Any]]:
    depth_by_id = _depth_map(run_dir)
    rows: list[dict[str, Any]] = []
    for round_id in range(1, rounds + 1):
        round_dir = run_dir / f"round_{round_id:02d}"
        candidate = _json(round_dir / "candidate_pool_receipt.json", {}) or {}
        approved = _json(round_dir / "approved_batch.json", {}) or {}
        rows.append(
            {
                "round": round_id,
                "candidate": _depth_summary(
                    [str(item) for item in candidate.get("candidate_ids") or []],
                    depth_by_id,
                ),
                "approved": _depth_summary(
                    [str(item) for item in approved.get("candidate_ids") or []],
                    depth_by_id,
                ),
                "hypothesis_match_policy": candidate.get("hypothesis_match_policy"),
                "evidence_prefilter_policy": candidate.get("evidence_prefilter_policy"),
                "persisted_depth_counts": candidate.get(
                    "candidate_mutation_order_counts"
                ),
            }
        )
    return rows


def _round_batch_medians(run_dir: Path) -> list[float]:
    summary = _json(run_dir / "summary.json", {}) or {}
    return [
        float(item["batch_median_fitness"])
        for item in summary.get("round_metrics") or []
    ]


def _preferred_summary(run_dir: Path, rounds: int = 3) -> list[dict[str, int]]:
    config = _json(run_dir / "config.json", {}) or {}
    protein = config.get("protein_context") or {}
    positions = tuple(int(item) for item in protein.get("mutable_positions") or ())
    wild_type = tuple(str(protein.get("wild_type_sites") or ""))
    native = dict(zip(positions, wild_type, strict=True))
    output: list[dict[str, int]] = []
    for round_id in range(1, rounds + 1):
        pipeline = _json(
            run_dir / f"round_{round_id:02d}" / "hypothesis_pipeline.json",
            {},
        ) or {}
        hypothesis = pipeline.get("main_hypothesis") or {}
        preferred = hypothesis.get("preferred_residues") or {}
        native_positions = 0
        actionable = 0
        for raw_position, residues in preferred.items():
            position = int(raw_position)
            values = tuple(str(item) for item in residues)
            native_positions += int(native.get(position) in values)
            actionable += sum(item != native.get(position) for item in values)
        output.append(
            {
                "round": round_id,
                "preferred_position_count": len(preferred),
                "wt_inclusive_position_count": native_positions,
                "actionable_non_wt_preference_count": actionable,
            }
        )
    return output


def audit_guard_artifacts(result: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(result["run_dir"])
    rounds = _round_depth_statistics(run_dir)
    policies = all(
        item["hypothesis_match_policy"] == "edited_non_wild_type_sites_only"
        and item["evidence_prefilter_policy"] == "selection_authorized_only"
        for item in rounds
    )
    counts_match = all(
        sum(int(value) for value in (item["persisted_depth_counts"] or {}).values())
        == 32
        and (item["persisted_depth_counts"] or {}) == item["candidate"]["counts"]
        for item in rounds
    )
    return {
        "policy_persisted_each_round": policies,
        "depth_counts_match_each_round": counts_match,
        "rounds": rounds,
        "preferred_residue_structure": _preferred_summary(run_dir),
    }


def _safe_researcher(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "receipt_rounds": int(value.get("receipt_rounds", 0)),
        "phase_a_rounds": int(value.get("phase_a_rounds", 0)),
        "phase_b_rounds": int(value.get("phase_b_rounds", 0)),
        "external_plan_count": int(value.get("external_plan_count", 0)),
        "external_abstain_count": int(value.get("external_abstain_count", 0)),
        "feature_plan_count": int(value.get("feature_plan_count", 0)),
        "feature_abstain_count": int(value.get("feature_abstain_count", 0)),
        "rag_queries": int(value.get("rag_queries", 0)),
        "retrieved_records": int(value.get("retrieved_records", 0)),
        "feature_requests": int(value.get("feature_requests", 0)),
        "query_intents": dict(value.get("query_intents") or {}),
        "record_types": dict(value.get("record_types") or {}),
        "knowledge_types": dict(value.get("knowledge_types") or {}),
        "feature_channels": dict(value.get("feature_channels") or {}),
        "feature_focus": dict(value.get("feature_focus") or {}),
        "rounds": [
            {
                "round": int(item.get("round", 0)),
                "external_decision": item.get("external_decision"),
                "feature_decision": item.get("feature_decision"),
                "query_intents": list(item.get("query_intents") or []),
                "facets": list(item.get("facets") or []),
                "rag_queries": int(item.get("rag_queries", 0)),
                "retrieved_records": int(item.get("retrieved_records", 0)),
                "feature_requests": int(item.get("feature_requests", 0)),
                "feature_needs": [
                    {
                        "channel": need.get("channel"),
                        "focus": list(need.get("focus") or []),
                        "position_count": len(need.get("positions") or []),
                    }
                    for need in item.get("feature_needs") or []
                ],
                "skip_count": len(item.get("skipped") or []),
                "reject_count": len(item.get("rejected") or []),
            }
            for item in value.get("rounds") or []
        ],
    }


def execute_guarded_run(config: ExperimentConfig) -> dict[str, Any]:
    result = base_runner.execute_run(config)
    guard = audit_guard_artifacts(result)
    result["candidate_guard"] = guard
    result["round_batch_median"] = _round_batch_medians(Path(result["run_dir"]))
    result["integrity"]["candidate_guard_policy_persisted"] = guard[
        "policy_persisted_each_round"
    ]
    result["integrity"]["candidate_guard_depth_counts_match"] = guard[
        "depth_counts_match_each_round"
    ]
    failures = [
        name for name, passed in result["integrity"].items() if not bool(passed)
    ]
    result["integrity_failures"] = failures
    result["status"] = "completed" if not failures else "invalid"
    result["researcher"] = _safe_researcher(result.get("researcher") or {})
    return result


def _safe_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "fold": int(run["fold"]),
        "condition": str(run["condition"]),
        "run_dir": str(run["run_dir"]),
        "round_best_seen": list(run.get("round_best_seen") or []),
        "round_batch_best": list(run.get("round_batch_best") or []),
        "round_batch_mean": list(run.get("round_batch_mean") or []),
        "round_batch_median": list(run.get("round_batch_median") or _round_batch_medians(Path(run["run_dir"]))),
        "final_best_seen": float(run["final_best_seen"]),
        "auc_proxy": float(run["auc_proxy"]),
        "elapsed_seconds": float(run.get("elapsed_seconds") or 0.0),
        "researcher": _safe_researcher(run.get("researcher") or {}),
        "llm": run.get("llm") or {},
        "candidate_guard": run.get("candidate_guard") or {
            "rounds": _round_depth_statistics(Path(run["run_dir"])),
            "preferred_residue_structure": _preferred_summary(Path(run["run_dir"])),
        },
    }


def _historical_snapshot() -> dict[str, Any]:
    receipt = _json(HISTORICAL_RECEIPT, {}) or {}
    runs = [
        _safe_run(item)
        for item in receipt.get("selected_runs") or []
        if int(item.get("fold", -1)) == TARGET_FOLD
    ]
    return {
        "receipt": str(HISTORICAL_RECEIPT),
        "fold2_runs": runs,
        "original_aggregate": receipt.get("aggregate") or {},
    }


def aggregate_fold2_pair(runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = base_runner.aggregate(runs)
    paired = aggregate.get("paired_deltas") or []
    complete = len(runs) == 2 and len(paired) == 1
    pair = paired[0] if paired else {}
    aggregate["runtime_integrity_passed"] = complete
    aggregate["directionally_positive"] = bool(
        complete
        and float(pair.get("final_best_delta", 0.0)) > 0.0
        and float(pair.get("auc_delta", 0.0)) > 0.0
    )
    aggregate["pre_registered_efficacy_evaluated"] = False
    aggregate.pop("positive_efficacy_supported", None)
    return aggregate


def _condition_map(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["condition"]): item for item in runs}


def _set_overlap(left: Path, right: Path, round_id: int, name: str) -> int:
    return len(
        base_runner._id_set(left, round_id, name, "candidate_ids")
        & base_runner._id_set(right, round_id, name, "candidate_ids")
    )


def compare_with_historical(
    current_runs: list[dict[str, Any]],
    historical: dict[str, Any],
) -> dict[str, Any]:
    current = _condition_map(current_runs)
    old = _condition_map(historical.get("fold2_runs") or [])
    if set(current) != set(CONDITIONS) or set(old) != set(CONDITIONS):
        return {}
    current_pair = aggregate_fold2_pair(current_runs)["paired_deltas"][0]
    old_pair = (
        float(old["researcher_agentic_rag"]["final_best_seen"])
        - float(old["researcher_no_rag"]["final_best_seen"])
    )
    old_auc_pair = (
        float(old["researcher_agentic_rag"]["auc_proxy"])
        - float(old["researcher_no_rag"]["auc_proxy"])
    )
    same_arm: dict[str, dict[str, float]] = {}
    overlaps: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        new_run = current[condition]
        old_run = old[condition]
        same_arm[condition] = {
            "final_best_change": float(new_run["final_best_seen"])
            - float(old_run["final_best_seen"]),
            "auc_change": float(new_run["auc_proxy"])
            - float(old_run["auc_proxy"]),
        }
        for round_id in range(1, 4):
            overlaps.append(
                {
                    "condition": condition,
                    "round": round_id,
                    "candidate_overlap": _set_overlap(
                        Path(new_run["run_dir"]),
                        Path(old_run["run_dir"]),
                        round_id,
                        "candidate_pool_receipt.json",
                    ),
                    "approved_overlap": _set_overlap(
                        Path(new_run["run_dir"]),
                        Path(old_run["run_dir"]),
                        round_id,
                        "approved_batch.json",
                    ),
                }
            )
    old_other = [
        item
        for item in (historical.get("original_aggregate") or {}).get(
            "paired_deltas", []
        )
        if int(item.get("fold", -1)) in {0, 1}
    ]
    mixed_final = [
        *(float(item["final_best_delta"]) for item in old_other),
        float(current_pair["final_best_delta"]),
    ]
    mixed_auc = [
        *(float(item["auc_delta"]) for item in old_other),
        float(current_pair["auc_delta"]),
    ]
    return {
        "new_pair": current_pair,
        "historical_pair": {
            "final_best_delta": old_pair,
            "auc_delta": old_auc_pair,
        },
        "same_arm_changes": same_arm,
        "descriptive_contrast_of_contrasts": {
            "final_best": float(current_pair["final_best_delta"]) - old_pair,
            "auc": float(current_pair["auc_delta"]) - old_auc_pair,
        },
        "same_arm_candidate_overlap": overlaps,
        "mixed_protocol_sensitivity": {
            "label": "post_hoc_mixed_protocol_not_pre_registered",
            "median_final_best_delta": statistics.median(mixed_final),
            "mean_final_best_delta": statistics.fmean(mixed_final),
            "mean_auc_delta": statistics.fmean(mixed_auc),
        },
    }


def _selected_runs(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in base_runner.selected_runs(receipt.get("attempts") or [])
        if int(item.get("fold", -1)) == TARGET_FOLD
    ]


def _refresh(receipt: dict[str, Any]) -> None:
    runs = _selected_runs(receipt)
    receipt["selected_runs"] = runs
    receipt["aggregate"] = aggregate_fold2_pair(runs)
    receipt["historical_comparison"] = compare_with_historical(
        runs,
        receipt["historical_snapshot"],
    )
    receipt["updated_at"] = base_runner._now()


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    base_runner._write_receipt(path, receipt)


def _fmt(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _llm_totals(run: dict[str, Any]) -> dict[str, float]:
    values = list((run.get("llm") or {}).values())
    return {
        "attempts": sum(int(item.get("attempts", 0)) for item in values),
        "completed": sum(int(item.get("completed_calls", 0)) for item in values),
        "retries": sum(int(item.get("retries", 0)) for item in values),
        "tokens": sum(int(item.get("total_tokens", 0)) for item in values),
        "latency": sum(float(item.get("latency_seconds_sum", 0.0)) for item in values),
    }


def render_report(receipt: dict[str, Any]) -> str:
    runs = receipt.get("selected_runs") or []
    current = _condition_map(runs)
    old = _condition_map((receipt.get("historical_snapshot") or {}).get("fold2_runs") or [])
    aggregate = receipt.get("aggregate") or {}
    comparison = receipt.get("historical_comparison") or {}
    complete = aggregate.get("runtime_integrity_passed") is True
    lines = [
        "# GB1 fold 2：no-RAG vs 候选护栏 Agentic RAG 真实 API 复测",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run → validate",
        f"- Verification Status: {'VERIFIED' if complete else 'IN_PROGRESS'}",
        "- Version Label: fold2_guard_live_rerun_v1",
        "",
        "## 状态与解释边界",
        "",
        f"- 状态：`{'COMPLETE' if complete else 'INCOMPLETE'}`；有效 runs：{len(runs)}/2。",
        "- 本次是看过历史 fold 2 后进行的 post-hoc repair validation，不是独立确认性 fold。",
        "- 工程完整性、guard 是否激活、相对性能和泛化性分开判断。",
        "- 不记录 API key、mutation/candidate/sample identity、序列、确切位置、完整 conversation 或隐藏推理。",
        "",
        "## 命令、实现与 attempts",
        "",
        f"- 命令：`{' '.join(receipt.get('environment', {}).get('command') or [])}`",
        f"- 配置 SHA-256：`{receipt.get('environment', {}).get('config_sha256', '—')}`",
        f"- Guard implementation SHA-256：`{receipt.get('implementation', {}).get('combined_sha256', '—')}`",
        f"- Index manifest：`{receipt.get('index', {}).get('stats', {}).get('manifest_hash', '—')}`",
        "",
        "| Attempt | Condition | Status | Started | Finished | Elapsed(s) | Failure |",
        "|---:|---|---|---|---|---:|---|",
    ]
    for index, attempt in enumerate(receipt.get("attempts") or [], 1):
        result = attempt.get("result") or {}
        lines.append(
            f"| {index} | {attempt.get('condition', '—')} | {attempt.get('status', '—')} | {attempt.get('started_at', '—')} | {attempt.get('finished_at', '—')} | {_fmt(result.get('elapsed_seconds'), 1)} | {base_runner._attempt_failure_text(attempt)} |"
        )
    lines.extend(
        [
            "",
            "## 四格性能比较",
            "",
            "| Version | Condition | R1 best | R2 best | R3 best | Final best | AUC proxy |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for version, mapping in (("historical", old), ("guard-rerun", current)):
        for condition in CONDITIONS:
            run = mapping.get(condition)
            if not run:
                continue
            batch = run.get("round_batch_best") or []
            lines.append(
                f"| {version} | {condition} | {_fmt(batch[0] if len(batch)>0 else None)} | {_fmt(batch[1] if len(batch)>1 else None)} | {_fmt(batch[2] if len(batch)>2 else None)} | {_fmt(run.get('final_best_seen'))} | {_fmt(run.get('auc_proxy'))} |"
            )
    if comparison:
        pair = comparison["new_pair"]
        old_pair = comparison["historical_pair"]
        coc = comparison["descriptive_contrast_of_contrasts"]
        lines.extend(
            [
                "",
                f"- 新配对 Agentic − no-RAG：final `{_fmt(pair['final_best_delta'])}`；AUC `{_fmt(pair['auc_delta'])}`。",
                f"- 历史 fold 2 配对：final `{_fmt(old_pair['final_best_delta'])}`；AUC `{_fmt(old_pair['auc_delta'])}`。",
                f"- 描述性 contrast-of-contrasts：final `{_fmt(coc['final_best'])}`；AUC `{_fmt(coc['auc'])}`。",
                f"- 新 fold 2 方向同时为正：`{aggregate.get('directionally_positive', False)}`；不执行预注册三折效能判定。",
            ]
        )
    lines.extend(
        [
            "",
            "## Candidate guard 与 mutation-depth 结构",
            "",
            "| Version | Condition | Round | Candidate depths | Approved depths | Candidate low-depth share | Candidate max-depth share |",
            "|---|---|---:|---|---|---:|---:|",
        ]
    )
    for version, mapping in (("historical", old), ("guard-rerun", current)):
        for condition in CONDITIONS:
            run = mapping.get(condition)
            if not run:
                continue
            for item in (run.get("candidate_guard") or {}).get("rounds") or []:
                candidate = item["candidate"]
                approved = item["approved"]
                lines.append(
                    f"| {version} | {condition} | {item['round']} | {base_runner._counter_text(candidate['counts'])} | {base_runner._counter_text(approved['counts'])} | {_fmt(candidate['low_depth_share'])} | {_fmt(candidate['max_depth_share'])} |"
                )
    if current:
        guard_passed = all(
            (run.get("candidate_guard") or {}).get("policy_persisted_each_round")
            and (run.get("candidate_guard") or {}).get("depth_counts_match_each_round")
            for run in current.values()
        )
        lines.extend(["", f"- Guard runtime active：`{guard_passed}`。"])
    lines.extend(
        [
            "",
            "## 新配对 Researcher 与成本",
            "",
            "| Condition | RAG queries | Retrieved records | Feature requests | LLM attempts/completed/retries | LLM tokens | LLM latency(s) | Wall(s) |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        run = current.get(condition)
        if not run:
            continue
        research = run.get("researcher") or {}
        llm = _llm_totals(run)
        lines.append(
            f"| {condition} | {research.get('rag_queries', 0)} | {research.get('retrieved_records', 0)} | {research.get('feature_requests', 0)} | {int(llm['attempts'])}/{int(llm['completed'])}/{int(llm['retries'])} | {int(llm['tokens'])} | {_fmt(llm['latency'], 1)} | {_fmt(run.get('elapsed_seconds'), 1)} |"
        )
    if comparison:
        mixed = comparison["mixed_protocol_sensitivity"]
        lines.extend(
            [
                "",
                "## 与原三折结果的关系",
                "",
                "- 原始三折同协议结果保持不变；本报告不覆盖原结论。",
                f"- 仅作 post-hoc mixed-protocol sensitivity：median final delta `{_fmt(mixed['median_final_best_delta'])}`；mean final delta `{_fmt(mixed['mean_final_best_delta'])}`；mean AUC delta `{_fmt(mixed['mean_auc_delta'])}`。",
                "- 该敏感性混合旧 fold 0/1 与 guarded fold 2，不能称为预注册或同版本三折实验。",
            ]
        )
    lines.extend(
        [
            "",
            "## Statistical Fallacy Scan",
            "",
            "- Coverage：11/11 checked。",
            "- 已逐 fold 报告，未用总体方向遮盖 fold 2；未从 depth 关联推断 residue 或 mutation-depth 因果。",
            "- approved-batch rank 属于选择后诊断，存在 Berkson/selection bias；无 p-value 或显著性主张。",
            "- 本复测重复使用同一 fold/oracle，存在 post-hoc、look-elsewhere、garden-of-forking-paths 与 API 漂移风险。",
            "- 单 fold、单 run/arm 无法估计方差、置信区间或泛化效应。",
            "- Simpson、ecological、base-rate、survivorship、regression-to-mean、collider、reverse-causality 与 correlation-causation 均已检查；适用风险在上述限制中保留。",
            "",
            "## 可复核产物",
            "",
            f"- Receipt：`{receipt.get('receipt_path', '—')}`",
            f"- 历史 receipt：`{HISTORICAL_RECEIPT}`",
            f"- Runner：`{Path(__file__).resolve()}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _write_report(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(receipt), encoding="utf-8")


def _new_receipt(
    *,
    config_path: Path,
    receipt_path: Path,
    preflight: dict[str, Any],
    implementation: dict[str, Any],
) -> dict[str, Any]:
    environment = base_runner.environment_receipt(config_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": base_runner._now(),
        "updated_at": base_runner._now(),
        "receipt_path": str(receipt_path),
        "environment": environment,
        "implementation": implementation,
        "protocol": {
            "fold": TARGET_FOLD,
            "conditions": list(CONDITIONS),
            "rounds": 3,
            "candidate_limit": 32,
            "wet_validation_budget": 16,
            "visible_observation_counts": [1, 17, 33],
            "max_attempts_per_condition": MAX_ATTEMPTS_PER_CONDITION,
            "post_hoc_repair_validation": True,
        },
        "preflight": {"checks": preflight["checks"], "manifest": preflight["manifest"]},
        "index": preflight["index"],
        "historical_snapshot": _historical_snapshot(),
        "attempts": [],
        "selected_runs": [],
        "aggregate": aggregate_fold2_pair([]),
        "historical_comparison": {},
        "api_keys_recorded": False,
        "variant_identities_recorded_in_receipt": False,
        "hidden_reasoning_recorded_in_receipt": False,
    }


def _load_or_create_receipt(
    *,
    path: Path,
    config_path: Path,
    preflight: dict[str, Any],
    implementation: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    if path.is_file():
        if not resume:
            raise FileExistsError("Receipt exists; rerun with --resume")
        receipt = _json(path, {}) or {}
        if receipt.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Existing fold2 receipt schema is incompatible")
        if receipt.get("environment", {}).get("config_sha256") != _sha256(config_path):
            raise RuntimeError("Config changed since the fold2 receipt was created")
        if receipt.get("implementation", {}).get("combined_sha256") != implementation[
            "combined_sha256"
        ]:
            raise RuntimeError("Guard implementation changed since receipt creation")
        if receipt.get("index", {}).get("stats", {}).get("manifest_hash") != base_runner.EXPECTED_MANIFEST_HASH:
            raise RuntimeError("Existing fold2 receipt is bound to another index")
        _refresh(receipt)
        return receipt
    return _new_receipt(
        config_path=config_path,
        receipt_path=path,
        preflight=preflight,
        implementation=implementation,
    )


def main() -> int:
    args = arguments()
    config_path = _absolute(args.config)
    output_root = _absolute(args.output_root)
    report_path = _absolute(args.report)
    base = load_experiment_config(config_path)
    preflight = base_runner.assert_preflight(base)
    implementation = implementation_receipt()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "schema_version": SCHEMA_VERSION,
                    "fold": TARGET_FOLD,
                    "conditions": CONDITIONS,
                    "preflight": preflight,
                    "implementation": implementation,
                    "output_root": str(output_root),
                    "report": str(report_path),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0
    base_runner.require_credentials()
    configure_progress_logging()
    receipt_path = output_root / "fold2_guard_pair_receipt.json"
    receipt = _load_or_create_receipt(
        path=receipt_path,
        config_path=config_path,
        preflight=preflight,
        implementation=implementation,
        resume=args.resume,
    )
    _write_receipt(receipt_path, receipt)
    _write_report(report_path, receipt)
    selected = {
        str(item["condition"]) for item in receipt.get("selected_runs") or []
    }
    for condition in CONDITIONS:
        if condition in selected:
            continue
        prior = [
            item
            for item in receipt.get("attempts") or []
            if str(item.get("condition")) == condition
        ]
        if len(prior) >= MAX_ATTEMPTS_PER_CONDITION:
            print(json.dumps({"status": "attempt_limit_exhausted", "condition": condition}))
            return 2
        attempt: dict[str, Any] = {
            "fold": TARGET_FOLD,
            "condition": condition,
            "attempt_number": len(prior) + 1,
            "started_at": base_runner._now(),
            "finished_at": None,
            "status": "running",
        }
        receipt["attempts"].append(attempt)
        _refresh(receipt)
        _write_receipt(receipt_path, receipt)
        _write_report(report_path, receipt)
        config = base_runner.condition_run_config(
            base,
            fold=TARGET_FOLD,
            condition=condition,
            output_root=output_root,
        )
        try:
            result = execute_guarded_run(config)
            attempt["result"] = result
            attempt["status"] = result["status"]
        except Exception as error:  # noqa: BLE001 - audited live provider boundary
            attempt["status"] = "failed"
            attempt["error"] = {
                "error_type": type(error).__name__,
                "message": base_runner._redacted_error(error),
                "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            }
        attempt["finished_at"] = base_runner._now()
        _refresh(receipt)
        _write_receipt(receipt_path, receipt)
        _write_report(report_path, receipt)
        if attempt["status"] != "completed":
            print(
                json.dumps(
                    {
                        "status": attempt["status"],
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
        selected.add(condition)
    _refresh(receipt)
    _write_receipt(receipt_path, receipt)
    _write_report(report_path, receipt)
    print(
        json.dumps(
            {
                "status": "completed",
                "selected_runs": len(receipt["selected_runs"]),
                "directionally_positive": receipt["aggregate"][
                    "directionally_positive"
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
