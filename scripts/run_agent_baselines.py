#!/usr/bin/env python3
"""Run native-client campaigns and pair RAG vs no-RAG knowledge agents."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.reporting import aggregate_runs
from fitness_agents.reporting.aggregate import infer_condition
from fitness_agents.utils.progress import add_logging_arguments, configure_from_args

PRESETS: dict[str, dict[str, str]] = {
    "al96": {
        "random": "configs/experiments/random_al96.yaml",
        "fitness_direct": "configs/experiments/fitness_direct_al96.yaml",
        "llm_agent": "configs/experiments/llm_agent_al96.yaml",
        "knowledge_agent": "configs/experiments/knowledge_agent_al96.yaml",
        "knowledge_agent_rag": "configs/experiments/knowledge_agent_al96_rag.yaml",
        "knowledge_agent_qwen_rag": "configs/experiments/knowledge_agent_qwen_al96.yaml",
    },
    "demo": {
        "random": "configs/experiments/random.yaml",
        "fitness_direct": "configs/experiments/fitness_direct.yaml",
        "llm_agent": "configs/experiments/llm_agent.yaml",
        "knowledge_agent": "configs/experiments/knowledge_agent.yaml",
        "knowledge_agent_rag": "configs/experiments/knowledge_agent_rag.yaml",
    },
}
COMPARISON_SETS: dict[str, tuple[str, ...]] = {
    "rag": ("knowledge_agent", "knowledge_agent_rag"),
    "agents": ("llm_agent", "knowledge_agent"),
    "agents_rag": ("llm_agent", "knowledge_agent", "knowledge_agent_rag"),
    "llm_vs_qwen_rag": ("llm_agent", "knowledge_agent_qwen_rag"),
    "agents_qwen_rag": ("llm_agent", "knowledge_agent", "knowledge_agent_qwen_rag"),
    "full": (
        "random",
        "fitness_direct",
        "llm_agent",
        "knowledge_agent",
        "knowledge_agent_rag",
        "knowledge_agent_qwen_rag",
    ),
}
DEFAULT_MODES = "random,fitness_direct,knowledge_agent"
SUMMARY_KEYS = {
    "run_id",
    "mode",
    "seed",
    "round_metrics",
    "final_prediction_metrics",
    "queries_used",
    "data_source",
}


def _parse_csv_ints(value: str | None) -> list[int | None]:
    if value is None or value.strip().lower() in {"", "config"}:
        return [None]
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parsed) != len(set(parsed)) or any(item < 0 for item in parsed):
        raise ValueError("Values must be unique non-negative integers")
    return list(parsed)


def _parse_csv_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def _load_job(
    *, preset: str, mode: str, seed: int | None, fold: int | None,
    rounds: int | None, budget: int | None,
) -> Any:
    overrides: dict[str, Any] = {
        "run_label": f"{preset}-{mode}",
        "condition": mode,
    }
    if seed is not None:
        overrides["seed"] = seed
    if rounds is not None:
        overrides["rounds"] = rounds
    if budget is not None:
        overrides["budget_per_round"] = budget
    config = load_experiment_config(project_root() / PRESETS[preset][mode], overrides=overrides)
    if fold is not None:
        config = replace(config, task=replace(config.task, fold_index=fold))
    return config


def _job_id(public: dict[str, Any]) -> str:
    fold = public["fold_index"]
    fold_label = f"f{int(fold):02d}" if fold is not None else "fxx"
    return f"{public['condition']}-s{public['seed']}-{fold_label}"


def _job_public(mode: str, config: Any) -> dict[str, Any]:
    local = config.knowledge.local_knowledge
    local_enabled = bool(local.enabled and config.knowledge_enabled)
    return {
        "mode": mode,
        "condition": config.condition or mode,
        "seed": config.seed,
        "fold_index": config.task.fold_index,
        "rounds": config.rounds,
        "budget_per_round": config.budget_per_round,
        "candidate_limit": config.candidate_limit,
        "provider": config.llm.provider,
        "fitness_model": config.model.name,
        "selection_driver": config.generation.selection_driver,
        "generation_uses_fitness_predictors": (
            config.generation.use_fitness_predictors
        ),
        "dry_validation_enabled": config.validation.enabled,
        "knowledge_enabled": config.knowledge_enabled,
        "local_knowledge_enabled": local_enabled,
        "allow_remote_context": bool(local.allow_remote_context),
        "rag_in_remote_context": bool(local_enabled and local.allow_remote_context),
        "max_tool_calls": config.kg_interaction.max_tool_calls,
    }


def _job_command(
    *,
    python_executable: str,
    preset: str,
    mode: str,
    config: Any,
    rounds: int | None,
    budget: int | None,
) -> tuple[str, ...]:
    root = project_root()
    command = [
        python_executable,
        "-m",
        "fitness_agents.cli",
        str(root / PRESETS[preset][mode]),
        "--seed",
        str(config.seed),
        "--fold-index",
        str(config.task.fold_index),
        "--run-label",
        str(config.run_label),
        "--condition",
        str(config.condition or mode),
    ]
    if rounds is not None:
        command.extend(["--rounds", str(rounds)])
    if budget is not None:
        command.extend(["--budget-per-round", str(budget)])
    local = config.knowledge.local_knowledge
    if local.enabled and config.knowledge_enabled:
        job_stem = f"{config.condition or mode}-s{config.seed}-f{config.task.fold_index:02d}"
        if local.retrieval.embedding_backend == "api":
            overlay_path = (
                root / "artifacts" / "local_knowledge" / "overlays" / f"{job_stem}.sqlite"
            )
            command.extend(["--local-knowledge-overlay", str(overlay_path)])
        else:
            index_path = root / "artifacts" / "local_knowledge" / f"{job_stem}.sqlite"
            command.extend(["--local-knowledge-index", str(index_path)])
    return tuple(command)


def _load_imported_summaries(paths: Sequence[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "results" in payload:
            for item in payload.get("results", ()):
                if item.get("status") not in {"completed", "imported"} or not item.get(
                    "summary"
                ):
                    continue
                summary = dict(item["summary"])
                job = item.get("job") or {}
                summary["condition"] = infer_condition(
                    summary, job_mode=job.get("mode") or job.get("condition")
                )
                summaries.append(summary)
            continue
        if isinstance(payload, dict) and payload.get("run_id"):
            payload = dict(payload)
            payload["condition"] = infer_condition(payload)
            summaries.append(payload)
    return summaries


def _discover_summaries(root: Path) -> list[Path]:
    return sorted(root.glob("*/summary.json")) if root.is_dir() else []


def _matches_requested(
    summary: dict[str, Any], *, seeds: list[int | None], folds: list[int | None]
) -> bool:
    concrete_seeds = [item for item in seeds if item is not None]
    if concrete_seeds and int(summary["seed"]) not in concrete_seeds:
        return False
    fold = summary.get("data_source", {}).get("fold_index")
    concrete_folds = [item for item in folds if item is not None]
    return not concrete_folds or fold in concrete_folds


@dataclass(frozen=True)
class BaselineJob:
    index: int
    public: dict[str, Any]
    command: tuple[str, ...]


@dataclass(frozen=True)
class BaselineJobResult:
    index: int
    public: dict[str, Any]
    status: str
    returncode: int
    stdout_log: str
    stderr_log: str
    summary: dict[str, Any] | None
    error: str | None = None


def _parse_summary(stdout: str, job: BaselineJob) -> dict[str, Any]:
    summary = json.loads(stdout)
    missing = SUMMARY_KEYS.difference(summary)
    if missing:
        raise ValueError(f"Campaign summary is missing keys: {sorted(missing)}")
    if int(summary.get("rounds_aborted") or 0) > 0:
        raise ValueError("Campaign aborted a round before completion")
    if not summary["round_metrics"]:
        raise ValueError("Campaign summary has no round metrics")
    fold = summary.get("data_source", {}).get("fold_index")
    if fold != job.public["fold_index"]:
        raise ValueError("Campaign summary reports a different fold")
    summary["condition"] = job.public["condition"]
    return summary


def _run_one_job(
    job: BaselineJob,
    *,
    project_dir: Path,
    output_dir: Path,
    timeout_seconds: float | None,
) -> BaselineJobResult:
    prefix = _job_id(job.public)
    stdout_path = output_dir / f"{prefix}.stdout.log"
    stderr_path = output_dir / f"{prefix}.stderr.log"
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
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        summary = None
        error = None
        status = "failed"
        if completed.returncode == 0:
            try:
                summary = _parse_summary(completed.stdout, job)
                status = "completed"
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as parse_error:
                error = f"Campaign output is not a valid summary: {parse_error}"
        else:
            error = f"Campaign exited with code {completed.returncode}"
        return BaselineJobResult(
            index=job.index,
            public=job.public,
            status=status,
            returncode=completed.returncode,
            stdout_log=str(stdout_path),
            stderr_log=str(stderr_path),
            summary=summary,
            error=error,
        )
    except subprocess.TimeoutExpired as timeout_error:
        stdout = timeout_error.stdout or ""
        stderr = timeout_error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return BaselineJobResult(
            index=job.index,
            public=job.public,
            status="timeout",
            returncode=124,
            stdout_log=str(stdout_path),
            stderr_log=str(stderr_path),
            summary=None,
            error=f"Campaign exceeded timeout of {timeout_seconds} seconds",
        )


def run_baseline_jobs(
    jobs: list[BaselineJob],
    *,
    max_parallel: int,
    project_dir: Path,
    output_dir: Path,
    timeout_seconds: float | None = None,
) -> list[BaselineJobResult]:
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    if not jobs:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[BaselineJobResult] = []
    workers = min(max_parallel, len(jobs))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent-baseline") as pool:
        future_by_job = {
            pool.submit(
                _run_one_job,
                job,
                project_dir=project_dir,
                output_dir=output_dir,
                timeout_seconds=timeout_seconds,
            ): job
            for job in jobs
        }
        for future in as_completed(future_by_job):
            result = future.result()
            results.append(result)
            identity = _job_id(result.public)
            print(
                f"{identity} status={result.status} returncode={result.returncode}",
                flush=True,
            )
    return sorted(results, key=lambda item: item.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native-client agents on paired folds. Use --comparison rag "
            "to isolate external-knowledge RAG vs the same knowledge agent without RAG. "
            "Use --max-parallel to run isolated campaign processes concurrently."
        )
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="al96")
    parser.add_argument(
        "--comparison",
        choices=sorted(COMPARISON_SETS),
        help="Named mode set. Ignored when --modes is also passed.",
    )
    parser.add_argument("--modes", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--folds", default="config")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum concurrent campaign processes. 1 is sequential; 4 runs four jobs at a time.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument(
        "--import-reports",
        help="Comma-separated report.json files from earlier baseline runs",
    )
    parser.add_argument(
        "--import-summaries",
        help="Comma-separated summary.json files to reuse as paired conditions",
    )
    parser.add_argument(
        "--import-run-root",
        type=Path,
        help="Directory of campaign run folders; loads each */summary.json",
    )
    parser.add_argument(
        "--skip-imported-conditions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not rerun a condition/seed/fold that was imported (default: true)",
    )
    parser.add_argument("--dry-run", action="store_true")
    add_logging_arguments(parser)
    return parser.parse_args()


def _resolve_modes(args: argparse.Namespace) -> list[str]:
    if args.modes:
        return [item.strip() for item in args.modes.split(",") if item.strip()]
    if args.comparison:
        return list(COMPARISON_SETS[args.comparison])
    return [item.strip() for item in DEFAULT_MODES.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    configure_from_args(args)
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be at least 1")
    modes = _resolve_modes(args)
    unknown = set(modes).difference(PRESETS[args.preset])
    if unknown:
        raise SystemExit(f"Unknown modes: {sorted(unknown)}")
    seeds = _parse_csv_ints(args.seeds)
    folds = _parse_csv_ints(args.folds)
    imported_paths = [
        *_parse_csv_paths(args.import_reports),
        *_parse_csv_paths(args.import_summaries),
    ]
    if args.import_run_root:
        imported_paths.extend(_discover_summaries(args.import_run_root))
    missing = [str(path) for path in imported_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Import files do not exist: {missing}")
    imported = [
        summary
        for summary in _load_imported_summaries(imported_paths)
        if _matches_requested(summary, seeds=seeds, folds=folds)
    ]
    imported_keys = {
        (
            infer_condition(summary),
            int(summary["seed"]),
            summary.get("data_source", {}).get("fold_index"),
        )
        for summary in imported
    }
    jobs: list[BaselineJob] = []
    for seed in seeds:
        for fold in folds:
            for mode in modes:
                config = _load_job(
                    preset=args.preset, mode=mode, seed=seed, fold=fold,
                    rounds=args.rounds, budget=args.budget,
                )
                public = _job_public(mode, config)
                key = (public["condition"], public["seed"], public["fold_index"])
                if args.skip_imported_conditions and key in imported_keys:
                    continue
                command = _job_command(
                    python_executable=sys.executable,
                    preset=args.preset,
                    mode=mode,
                    config=config,
                    rounds=args.rounds,
                    budget=args.budget,
                )
                public = {**public, "command": list(command)}
                jobs.append(BaselineJob(index=len(jobs), public=public, command=command))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or project_root() / "artifacts" / f"agent-baselines-{stamp}"
    schedule = {
        "comparison": args.comparison,
        "modes": modes,
        "max_parallel": args.max_parallel,
        "timeout_seconds": args.timeout_seconds,
        "imported_files": [str(path) for path in imported_paths],
        "imported_runs": [item["run_id"] for item in imported],
        "jobs": [job.public for job in jobs],
    }
    if args.dry_run:
        print(json.dumps({"output_dir": str(output_dir), **schedule}, indent=2, default=str))
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "schedule.json").write_text(
        json.dumps(schedule, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    timeout = args.timeout_seconds if args.timeout_seconds > 0 else None
    job_results = run_baseline_jobs(
        jobs,
        max_parallel=args.max_parallel,
        project_dir=project_root(),
        output_dir=output_dir / "job_logs",
        timeout_seconds=timeout,
    )
    summaries, results = list(imported), []
    for summary in imported:
        results.append(
            {
                "status": "imported",
                "job": {
                    "mode": infer_condition(summary),
                    "condition": infer_condition(summary),
                    "seed": summary["seed"],
                    "fold_index": summary.get("data_source", {}).get("fold_index"),
                    "run_id": summary["run_id"],
                },
                "summary": summary,
            }
        )
    for item in job_results:
        record = {
            "status": item.status,
            "job": item.public,
            "stdout_log": item.stdout_log,
            "stderr_log": item.stderr_log,
            "returncode": item.returncode,
        }
        if item.summary is not None:
            record["summary"] = item.summary
            summaries.append(item.summary)
        if item.error:
            record["error"] = item.error
        results.append(record)
    aggregate = aggregate_runs(summaries, output_dir / "aggregate") if summaries else {}
    report = {
        "completed": sum(item["status"] == "completed" for item in results),
        "imported": sum(item["status"] == "imported" for item in results),
        "failed": sum(item["status"] not in {"completed", "imported"} for item in results),
        "max_parallel": args.max_parallel,
        "aggregate": {key: str(value) for key, value in aggregate.items()},
        "results": results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
