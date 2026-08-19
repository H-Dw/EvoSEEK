#!/usr/bin/env python3
"""Run and artifact-audit a declarative RAG/KG/feature/AL route matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.loop import run_campaign
from fitness_agents.reporting import aggregate_runs

ALLOWED_CHANNELS = ("physchem", "conservation", "structure")
FEATURE_RELATIONS = {
    "physchem": "HAS_PHYSCHEM_DELTA",
    "conservation": "HAS_EVOLUTIONARY_CONTEXT",
    "structure": "OCCURS_IN_ENVIRONMENT",
}
BASE_AUDIT_ITEMS = ("MutationEffectEstimate", "HAS_MUTATION", "ABOUT_MUTATION")


@dataclass(frozen=True)
class RouteSpec:
    route_id: str
    rag: bool
    channels: tuple[str, ...]
    active_learning: bool
    test_goal: str


@dataclass(frozen=True)
class RouteJob:
    index: int
    route: RouteSpec
    fold_index: int
    seed: int
    command: tuple[str, ...]


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return payload


def load_matrix(path: Path) -> tuple[Path, list[int], int, int, list[RouteSpec]]:
    raw = _read_yaml(path)
    if raw.get("schema_version") != "reasoning-route-matrix:v1":
        raise ValueError("Unsupported reasoning route matrix schema_version")
    root = project_root()
    base_path = Path(str(raw["base_config"]))
    base_path = base_path if base_path.is_absolute() else root / base_path
    folds = [int(item) for item in raw.get("folds", (0, 1, 2))]
    if not folds or len(folds) != len(set(folds)) or any(item < 0 for item in folds):
        raise ValueError("folds must be unique non-negative integers")
    routes = []
    for route_id, values in dict(raw.get("routes", {})).items():
        values = dict(values or {})
        channels = tuple(str(item) for item in values.get("channels", ()))
        if len(channels) != len(set(channels)) or set(channels).difference(ALLOWED_CHANNELS):
            raise ValueError(f"Route {route_id!r} has invalid or duplicate channels")
        routes.append(
            RouteSpec(
                route_id=str(route_id),
                rag=bool(values.get("rag", False)),
                channels=channels,
                active_learning=bool(values.get("active_learning", False)),
                test_goal=str(values.get("test_goal", "")).strip(),
            )
        )
    if not routes:
        raise ValueError("Route matrix must contain at least one route")
    return (
        base_path,
        folds,
        int(raw.get("seed", 11)),
        int(raw.get("max_parallel", 2)),
        routes,
    )


def _parse_csv(value: str | None, default: list[int] | list[str]) -> list[Any]:
    if value is None or value.strip().lower() in {"", "config"}:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def required_tool_calls(spec: RouteSpec, *, variant_limit: int) -> int:
    """Minimum planned LLM-KG calls so RAG/feature/KG steps are not sliced off."""
    count = 2  # hypothesis_context + query_assay_association
    if spec.channels:
        count += max(1, variant_limit)
    count += 1  # truncation audit
    if spec.rag:
        count += 2  # query_local_knowledge + query_structured_claims
    return count + 2  # explain_variant + compare_variants


def apply_route(config: Any, spec: RouteSpec, *, fold: int, seed: int, output_root: Path) -> Any:
    local = replace(
        config.knowledge.local_knowledge,
        enabled=spec.rag,
        index_path=(output_root.parent / "local_knowledge" / f"{spec.route_id}-f{fold:02d}.sqlite"),
        corpus_index_path=(
            output_root.parent / "local_knowledge" / f"{spec.route_id}-f{fold:02d}.sqlite"
        ),
    )
    knowledge = replace(
        config.knowledge,
        physchem="physchem" in spec.channels,
        conservation="conservation" in spec.channels,
        structure="structure" in spec.channels,
        kg=True,
        local_knowledge=local,
    )
    operators = [
        "hypothesis_context",
        "query_assay_association",
        "query_evidence_provenance",
    ]
    if spec.channels:
        operators.append("query_feature_bundle")
    operators.append("query_kg_truncation_audit")
    if spec.rag:
        operators.extend(("query_local_knowledge", "query_structured_claims"))
    operators.extend(("explain_variant", "compare_variants"))
    audit_items = (*BASE_AUDIT_ITEMS, *(FEATURE_RELATIONS[item] for item in spec.channels))
    variant_limit = config.kg_interaction.feature_variant_limit
    interaction = replace(
        config.kg_interaction,
        enabled=True,
        enabled_operators=tuple(operators),
        feature_tool_strategy="joint" if spec.channels else "context_only",
        # The runtime contract requires a non-empty tuple even in context_only mode.
        feature_channels=spec.channels or ("physchem",),
        truncation_audit_enabled=True,
        truncation_audit_items=tuple(audit_items),
        max_tool_calls=max(
            config.kg_interaction.max_tool_calls,
            required_tool_calls(spec, variant_limit=variant_limit),
        ),
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
        kg_interaction=interaction,
        generation=generation,
        active_learning=active_learning,
        output_root=output_root,
        condition=spec.route_id,
        run_label=f"GB1-reasoning-{spec.route_id}-f{fold:02d}",
    )


def _check(
    name: str, passed: bool, detail: Any = None, *, severity: str = "error"
) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "detail": detail}


def audit_run(
    summary: dict[str, Any], spec: RouteSpec, *, expected_fold: int | None = None
) -> dict[str, Any]:
    run_dir = Path(summary["run_dir"])
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    completion_path = run_dir / "completion_manifest.json"
    completion = (
        json.loads(completion_path.read_text(encoding="utf-8"))
        if completion_path.is_file()
        else {}
    )
    round_dirs = sorted(path for path in run_dir.glob("round_*") if path.is_dir())
    checks = [
        _check("campaign_finalized", summary.get("finalized") is True),
        _check("completion_manifest_present", completion_path.is_file()),
        _check("artifact_finalized", completion.get("artifact_finalized") is True),
        _check("run_status_completed", completion.get("run_status") == "completed"),
        _check(
            "experiment_status_completed",
            completion.get("experiment_status") == "completed",
        ),
        _check("formal_pass_eligible", completion.get("pass_eligible") is True),
        _check("condition_matches_route", summary.get("condition") == spec.route_id),
        _check(
            "fold_matches_schedule",
            expected_fold is None
            or summary.get("data_source", {}).get("fold_index") == expected_fold,
            summary.get("data_source", {}).get("fold_index"),
        ),
        _check("kg_sqlite_created", (run_dir / "structured_kg.sqlite").is_file()),
        _check("kg_edges_exported", (run_dir / "knowledge_graph_edges.json").is_file()),
        _check("llm_is_non_mock", config.get("llm_provider") != "mock", config.get("llm_provider")),
        _check(
            "llm_hypothesis_created",
            bool(state.get("hypotheses")),
            len(state.get("hypotheses", ())),
        ),
        _check("round_artifacts_present", bool(round_dirs), len(round_dirs)),
        _check(
            "budget_per_round_is_16",
            config.get("budget_per_round") == 16,
            config.get("budget_per_round"),
        ),
        _check("rounds_are_3", config.get("rounds") == 3, config.get("rounds")),
        _check(
            "rounds_aborted_is_zero",
            int(summary.get("rounds_aborted") or 0) == 0,
            summary.get("rounds_aborted"),
        ),
        _check(
            "completed_rounds_match_config",
            len(summary.get("round_metrics") or ()) == int(config.get("rounds") or 0),
            {
                "round_metrics": len(summary.get("round_metrics") or ()),
                "rounds": config.get("rounds"),
            },
        ),
        _check(
            "queries_match_completed_budget",
            int(summary.get("queries_used") or 0)
            == int(config.get("budget_per_round") or 0)
            * len(summary.get("round_metrics") or ()),
            {
                "queries_used": summary.get("queries_used"),
                "budget_per_round": config.get("budget_per_round"),
                "completed_rounds": len(summary.get("round_metrics") or ()),
            },
        ),
        _check(
            "critic_is_remote_llm",
            config.get("critic", {}).get("mode") == "remote"
            and config.get("critic", {}).get("provider") not in {None, "", "mock"},
            config.get("critic"),
        ),
    ]
    interaction_cfg = config.get("kg_interaction", {})
    expected_strategy = "joint" if spec.channels else "context_only"
    checks.append(
        _check(
            "feature_tool_strategy_matches",
            interaction_cfg.get("feature_tool_strategy") == expected_strategy,
            interaction_cfg.get("feature_tool_strategy"),
        )
    )
    quota_cfg = config.get("generation", {}).get("quota_allocation", {})
    checks.append(
        _check(
            "quota_allocation_matches_driver",
            bool(quota_cfg.get("enabled")) == (not spec.active_learning),
            quota_cfg,
        )
    )
    if not spec.active_learning:
        checks.append(
            _check(
                "quota_allocation_is_8_3_3_2",
                quota_cfg.get("quotas")
                == {
                    "hypothesis_target": 8,
                    "evidence_prior": 3,
                    "coverage_exploration": 3,
                    "matched_control": 2,
                },
                quota_cfg.get("quotas"),
            )
        )
    runtime = config.get("knowledge_runtime", {})
    provider_status = runtime.get("provider_status", {})
    local_runtime = runtime.get("local_knowledge", {})
    checks.append(_check("rag_runtime_matches", bool(local_runtime.get("enabled")) == spec.rag))
    checks.append(
        _check(
            "rag_allowed_in_scientist_context",
            (not spec.rag) or bool(local_runtime.get("scientist_context_allowed")),
            local_runtime.get("scientist_context_allowed"),
        )
    )
    for channel in ALLOWED_CHANNELS:
        expected = channel in spec.channels
        status = provider_status.get(channel, {}).get("status")
        checks.append(
            _check(
                f"provider_{channel}_matches",
                (status == "ready") if expected else (status == "disabled"),
                status,
            )
        )

    all_operators: set[str] = set()
    observed_channels: set[str] = set()
    channel_counts: dict[str, int] = {}
    truncation_reports = []
    completed_round_ids = {
        int(item.get("round_id"))
        for item in summary.get("round_metrics") or ()
        if item.get("round_id") is not None
    }
    for round_dir in round_dirs:
        interaction_path = round_dir / "kg_interaction.json"
        checks.append(_check(f"{round_dir.name}_kg_interaction", interaction_path.is_file()))
        try:
            round_index = int(round_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            round_index = None
        if round_index in completed_round_ids:
            checks.append(
                _check(f"{round_dir.name}_selection", (round_dir / "selection.csv").is_file())
            )
            checks.append(
                _check(
                    f"{round_dir.name}_approved_batch",
                    (round_dir / "approved_batch.json").is_file(),
                )
            )
        if interaction_path.is_file():
            interaction = json.loads(interaction_path.read_text(encoding="utf-8"))
            all_operators.update(str(pack.get("operator")) for pack in interaction.get("packs", ()))
        evidence_path = round_dir / "evidence_contract.json"
        if evidence_path.is_file():
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            observed_channels.update(evidence.get("channel_counts", {}))
            for channel, count in evidence.get("channel_counts", {}).items():
                channel_counts[str(channel)] = channel_counts.get(str(channel), 0) + int(count)
        structured_path = round_dir / "structured_kg_pre_design.json"
        structured = (
            json.loads(structured_path.read_text(encoding="utf-8"))
            if structured_path.is_file()
            else {}
        )
        checks.append(
            _check(
                f"{round_dir.name}_structured_kg_populated",
                int(structured.get("entity_count", 0)) > 0
                and int(structured.get("relation_count", 0)) > 0,
                {
                    "entity_count": structured.get("entity_count"),
                    "relation_count": structured.get("relation_count"),
                },
            )
        )
        rag_files = all(
            (round_dir / name).is_file()
            for name in ("local_rag_retrieval.json", "local_rag_evidence.json")
        )
        checks.append(_check(f"{round_dir.name}_rag_artifacts_match", rag_files == spec.rag))
        if spec.rag and rag_files:
            retrieval = json.loads(
                (round_dir / "local_rag_retrieval.json").read_text(encoding="utf-8")
            )
            chunk_count = len(retrieval.get("chunks", ()))
            checks.append(
                _check(f"{round_dir.name}_rag_returned_chunks", chunk_count > 0, chunk_count)
            )
        truncation_path = round_dir / "kg_truncation_audit.json"
        checks.append(_check(f"{round_dir.name}_truncation_audit", truncation_path.is_file()))
        if truncation_path.is_file():
            truncation_reports.append(json.loads(truncation_path.read_text(encoding="utf-8")))

    checks.append(
        _check(
            "kg_context_tool_called", "hypothesis_context" in all_operators, sorted(all_operators)
        )
    )
    checks.append(
        _check(
            "kg_assay_tool_called",
            "query_assay_association" in all_operators,
            sorted(all_operators),
        )
    )
    checks.append(
        _check("rag_tool_call_matches", ("query_local_knowledge" in all_operators) == spec.rag)
    )
    checks.append(
        _check(
            "structured_claims_tool_matches",
            ("query_structured_claims" in all_operators) == spec.rag,
            sorted(all_operators),
        )
    )
    checks.append(
        _check(
            "feature_bundle_call_matches",
            ("query_feature_bundle" in all_operators) == bool(spec.channels),
        )
    )
    for channel in ALLOWED_CHANNELS:
        checks.append(
            _check(
                f"evidence_{channel}_matches",
                (channel_counts.get(channel, 0) > 0) == (channel in spec.channels),
                channel_counts,
            )
        )
    expected_items = {*BASE_AUDIT_ITEMS, *(FEATURE_RELATIONS[item] for item in spec.channels)}
    audited_items = {
        str(entry.get("item"))
        for report in truncation_reports
        for entry in report.get("entries", ())
    }
    checks.append(
        _check(
            "truncation_keywords_audited",
            expected_items.issubset(audited_items),
            {"expected": sorted(expected_items), "audited": sorted(audited_items)},
        )
    )
    any_truncated = any(bool(report.get("any_truncated")) for report in truncation_reports)
    checks.append(_check("max_rows_truncation_detected", True, any_truncated, severity="info"))
    selection_driver = summary.get("selection_driver")
    checks.append(
        _check(
            "selection_driver_matches",
            selection_driver == ("active_learning" if spec.active_learning else "agent_uq"),
            selection_driver,
        )
    )
    al_files = all(
        (round_dir / "active_learning_posterior.json").is_file()
        and (round_dir / "active_learning_acquisition.json").is_file()
        for round_dir in round_dirs
    )
    checks.append(_check("active_learning_artifacts_match", al_files == spec.active_learning))
    failed = [item for item in checks if item["severity"] == "error" and not item["passed"]]
    return {
        "route_id": spec.route_id,
        "fold_index": summary.get("data_source", {}).get("fold_index"),
        "run_id": summary.get("run_id"),
        "run_dir": str(run_dir),
        "test_goal": spec.test_goal,
        "passed": not failed,
        "checks": checks,
        "notes": {
            "any_max_rows_truncation": any_truncated,
            "causal_claim": "Execution and context injection are audited; LLM reliance requires a separate prompt/evidence intervention test.",
        },
    }


def _run_job(job: RouteJob, *, root: Path, log_dir: Path, timeout: float | None) -> dict[str, Any]:
    prefix = f"{job.route.route_id}-f{job.fold_index:02d}-s{job.seed}"
    try:
        completed = subprocess.run(
            list(job.command),
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        (log_dir / f"{prefix}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (log_dir / f"{prefix}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            return {
                "index": job.index,
                "route": job.route.route_id,
                "fold_index": job.fold_index,
                "status": "failed",
                "returncode": completed.returncode,
            }
        summary = json.loads(completed.stdout)
        audit = audit_run(summary, job.route, expected_fold=job.fold_index)
        return {
            "index": job.index,
            "route": job.route.route_id,
            "fold_index": job.fold_index,
            "status": "passed" if audit["passed"] else "audit_failed",
            "returncode": 0,
            "summary": summary,
            "audit": audit,
        }
    except subprocess.TimeoutExpired:
        return {
            "index": job.index,
            "route": job.route.route_id,
            "fold_index": job.fold_index,
            "status": "timeout",
            "returncode": 124,
        }
    except (KeyError, ValueError, json.JSONDecodeError, OSError) as error:
        return {
            "index": job.index,
            "route": job.route.route_id,
            "fold_index": job.fold_index,
            "status": "audit_failed",
            "returncode": 0,
            "error": f"{type(error).__name__}: {error}",
        }


def _worker(args: argparse.Namespace) -> None:
    matrix_path = args.matrix.resolve()
    base_path, _folds, matrix_seed, _parallel, routes = load_matrix(matrix_path)
    by_id = {item.route_id: item for item in routes}
    if args.worker_route not in by_id:
        raise SystemExit(f"Unknown route {args.worker_route!r}")
    seed = matrix_seed if args.seed is None else args.seed
    config = apply_route(
        load_experiment_config(base_path),
        by_id[args.worker_route],
        fold=args.worker_fold,
        seed=seed,
        output_root=args.worker_output_root.resolve(),
    )
    print(json.dumps(run_campaign(config), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and audit RAG/KG/feature/AL routes")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=project_root() / "configs/experiments/gb1_reasoning_routes.matrix.yaml",
    )
    parser.add_argument("--routes", default="all", help="all or comma-separated route IDs")
    parser.add_argument("--folds", default="config", help="config or comma-separated fold indices")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-route", help=argparse.SUPPRESS)
    parser.add_argument("--worker-fold", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-root", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_route:
        _worker(args)
        return
    root = project_root()
    matrix_path = args.matrix if args.matrix.is_absolute() else root / args.matrix
    base_path, default_folds, matrix_seed, matrix_parallel, all_routes = load_matrix(matrix_path)
    route_ids = [item.route_id for item in all_routes]
    requested_routes = route_ids if args.routes == "all" else _parse_csv(args.routes, route_ids)
    if not requested_routes:
        raise SystemExit("At least one route must be selected")
    unknown = set(requested_routes).difference(route_ids)
    if unknown:
        raise SystemExit(f"Unknown routes: {sorted(unknown)}")
    folds = (
        default_folds
        if args.folds.strip().lower() == "config"
        else [int(item) for item in _parse_csv(args.folds, default_folds)]
    )
    if not folds or len(folds) != len(set(folds)) or any(fold < 0 for fold in folds):
        raise SystemExit("Folds must be unique non-negative integers")
    seed = matrix_seed if args.seed is None else args.seed
    max_parallel = matrix_parallel if args.max_parallel is None else args.max_parallel
    if max_parallel < 1:
        raise SystemExit("--max-parallel must be at least 1")
    config = load_experiment_config(base_path)
    manifest_path = (
        config.task.split_root / "manifest.public.json" if config.task.split_root else None
    )
    manifest = _read_yaml(manifest_path) if manifest_path and manifest_path.is_file() else None
    if manifest is not None:
        invalid_folds = [fold for fold in folds if fold >= int(manifest["n_folds"])]
        if invalid_folds:
            raise SystemExit(f"Fold indices exceed manifest bounds: {invalid_folds}")
        if manifest.get("strategy") != config.task.expected_split_strategy:
            raise SystemExit("Split manifest strategy differs from task configuration")
        if manifest.get("protocol_version") != config.task.expected_protocol_version:
            raise SystemExit("Split manifest protocol differs from task configuration")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or root / "artifacts" / f"reasoning-routes-{stamp}"
    selected = [item for item in all_routes if item.route_id in requested_routes]
    jobs = []
    for spec in selected:
        for fold in folds:
            command = (
                sys.executable,
                str(Path(__file__).resolve()),
                "--matrix",
                str(matrix_path.resolve()),
                "--worker-route",
                spec.route_id,
                "--worker-fold",
                str(fold),
                "--seed",
                str(seed),
                "--worker-output-root",
                str((output_dir / "runs").resolve()),
            )
            jobs.append(RouteJob(len(jobs), spec, fold, seed, command))
    schedule = {
        "schema_version": "reasoning-route-schedule:v1",
        "matrix": str(matrix_path.resolve()),
        "base_config": str(base_path.resolve()),
        "manifest": str(manifest_path) if manifest_path else None,
        "manifest_exists": bool(manifest_path and manifest_path.is_file()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if manifest_path and manifest_path.is_file()
        else None,
        "folds": folds,
        "seed": seed,
        "max_parallel": max_parallel,
        "jobs": [{**asdict(job), "route": asdict(job.route)} for job in jobs],
    }
    if args.dry_run:
        print(json.dumps(schedule, indent=2, ensure_ascii=False, default=str))
        return
    if not manifest_path or not manifest_path.is_file():
        raise SystemExit(
            f"Split manifest does not exist: {manifest_path}. Build it before running routes."
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    log_dir = output_dir / "job_logs"
    log_dir.mkdir()
    (output_dir / "schedule.json").write_text(
        json.dumps(schedule, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    timeout = args.timeout_seconds if args.timeout_seconds > 0 else None
    results = []
    with ThreadPoolExecutor(
        max_workers=min(max_parallel, len(jobs)), thread_name_prefix="reasoning-route"
    ) as pool:
        futures = {
            pool.submit(_run_job, job, root=root, log_dir=log_dir, timeout=timeout): job
            for job in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"route={result['route']} fold={result['fold_index']:02d} status={result['status']}",
                flush=True,
            )
    results.sort(key=lambda item: item["index"])
    summaries = [item["summary"] for item in results if item.get("summary")]
    aggregate = aggregate_runs(summaries, output_dir / "aggregate") if summaries else {}
    fold_integrity = []
    for fold in folds:
        fold_summaries = [
            item for item in summaries if item.get("data_source", {}).get("fold_index") == fold
        ]
        assignment_hashes = {
            item.get("data_source", {}).get("assignment_sha256") for item in fold_summaries
        }
        assignment_hashes.discard(None)
        actual_routes = {str(item.get("condition")) for item in fold_summaries}
        fold_integrity.append(
            {
                "fold_index": fold,
                "passed": len(fold_summaries) == len(selected)
                and actual_routes == set(requested_routes)
                and len(assignment_hashes) == 1,
                "expected_routes": requested_routes,
                "actual_routes": sorted(actual_routes),
                "assignment_sha256": sorted(assignment_hashes),
            }
        )
    global_failures = sum(not item["passed"] for item in fold_integrity)
    passed_jobs = sum(item["status"] == "passed" for item in results)
    failed_jobs = sum(item["status"] != "passed" for item in results)
    report = {
        "overall_passed": failed_jobs == 0 and global_failures == 0,
        "passed_jobs": passed_jobs,
        "failed_jobs": failed_jobs,
        "global_checks_failed": global_failures,
        "expected": len(jobs),
        "max_parallel": max_parallel,
        "paired_fold_integrity": fold_integrity,
        "aggregate": {key: str(value) for key, value in aggregate.items()},
        "results": results,
    }
    (output_dir / "route_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
