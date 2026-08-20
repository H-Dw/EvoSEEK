#!/usr/bin/env python3
"""Rerun random and fitness_direct campaigns with 16 wet-validated samples per round.

Each job runs 3 closed-loop rounds on GB1 AL96. Per round the campaign selects
exactly ``budget_per_round`` variants (default 16) for oracle / wet validation.

Fitness scoring uses the same Kermut/ESM-2 model as
``run_hierarchical_scientist.py``. Closed-pool jobs pin ``candidate_limit`` to
the GB1 32-candidate scoring budget; fitness_direct ranks that proposal pool,
and dry validation / oracle measurement only cover the selected batch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
from typing import Any, TextIO

from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.reporting import aggregate_runs
from fitness_agents.utils.progress import add_logging_arguments, configure_from_args

CONFIGS: dict[str, str] = {
    "random": "configs/experiments/random_al96_b16.yaml",
    "fitness_direct": "configs/experiments/fitness_direct_al96_b16.yaml",
}
HIERARCHICAL_CONFIG = "configs/experiments/hierarchical_scientist.deepseek.yaml"
KERMUT_MODEL = "kermut"
DEFAULT_MODES = ("random", "fitness_direct")
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
class CampaignJob:
    index: int
    mode: str
    seed: int
    fold_index: int
    rounds: int
    budget: int
    command: tuple[str, ...]


@dataclass(frozen=True)
class CampaignJobResult:
    job: CampaignJob
    status: str
    returncode: int
    stdout_log: str
    stderr_log: str
    summary: dict[str, Any] | None
    error: str | None = None


def _drain_stream(stream: TextIO, path: Path, echo: TextIO | None) -> str:
    chunks: list[str] = []
    with path.open("w", encoding="utf-8") as handle:
        for line in stream:
            handle.write(line)
            handle.flush()
            if echo is not None:
                echo.write(line)
                echo.flush()
            chunks.append(line)
    return "".join(chunks)


def _parse_csv_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer")
    if len(parsed) != len(set(parsed)) or any(item < 0 for item in parsed):
        raise ValueError("Values must be unique non-negative integers")
    return parsed


def _load_config(mode: str, *, seed: int, fold: int, rounds: int, budget: int):
    root = project_root()
    config = load_experiment_config(
        root / CONFIGS[mode],
        overrides={
            "seed": seed,
            "rounds": rounds,
            "budget_per_round": budget,
            "run_label": "al96-b16",
            "condition": mode,
        },
    )
    config = replace(config, task=replace(config.task, fold_index=fold))
    _assert_aligned_with_hierarchical(config)
    return config


@lru_cache(maxsize=1)
def hierarchical_evaluation_contract() -> dict[str, Any]:
    hierarchical = load_experiment_config(project_root() / HIERARCHICAL_CONFIG)
    return {
        "metrics": tuple(hierarchical.evaluation.metrics),
        "top_k": int(hierarchical.evaluation.top_k),
        "rounds": int(hierarchical.rounds),
        "budget_per_round": int(hierarchical.budget_per_round),
        "candidate_limit": int(hierarchical.candidate_limit),
        "final_test_model": hierarchical.model.name,
    }


def _assert_aligned_with_hierarchical(config: Any) -> None:
    contract = hierarchical_evaluation_contract()
    if tuple(config.evaluation.metrics) != contract["metrics"]:
        raise SystemExit(
            "evaluation.metrics must match hierarchical Scientist: "
            f"{list(contract['metrics'])}"
        )
    if int(config.evaluation.top_k) != contract["top_k"]:
        raise SystemExit(
            f"evaluation.top_k must be {contract['top_k']} to match hierarchical Scientist"
        )
    if config.model.name != KERMUT_MODEL:
        raise SystemExit("model_config must be Kermut/ESM-2")
    if contract["final_test_model"] != KERMUT_MODEL:
        raise SystemExit(
            "hierarchical Scientist must also use Kermut/ESM-2 as the fitness model"
        )
    if config.model.name != contract["final_test_model"]:
        raise SystemExit(
            "model_config must be the hierarchical final-test predictor "
            f"({contract['final_test_model']})"
        )
    if config.candidate_limit != contract["candidate_limit"]:
        raise SystemExit(
            "candidate_limit must match hierarchical Scientist "
            f"({contract['candidate_limit']})"
        )
    named_models = (
        config.model,
        *config.generation.predictor_models,
        *config.validation.predictor_models,
    )
    if any("onehot" in (item.name or "").casefold() for item in named_models):
        raise SystemExit("one-hot ensemble is not allowed in this baseline runner")


def _job_command(
    *,
    python_executable: str,
    mode: str,
    seed: int,
    fold: int,
    rounds: int,
    budget: int,
    quiet: bool,
    log_level: str | None,
) -> tuple[str, ...]:
    root = project_root()
    command = [
        python_executable,
        "-m",
        "fitness_agents.cli",
        str(root / CONFIGS[mode]),
        "--seed",
        str(seed),
        "--fold-index",
        str(fold),
        "--rounds",
        str(rounds),
        "--budget-per-round",
        str(budget),
        "--output-top-k",
        str(budget),
        "--run-label",
        "al96-b16",
        "--condition",
        mode,
    ]
    if quiet:
        command.append("--quiet")
    elif log_level:
        command.extend(["--log-level", log_level])
    return tuple(command)


def _job_id(job: CampaignJob) -> str:
    return f"{job.mode}-s{job.seed}-f{job.fold_index:02d}"


def _validate_summary(summary: dict[str, Any], job: CampaignJob) -> dict[str, Any]:
    missing = SUMMARY_KEYS.difference(summary)
    if missing:
        raise ValueError(f"Campaign summary is missing keys: {sorted(missing)}")
    if summary.get("mode") != job.mode:
        raise ValueError(f"Expected mode {job.mode}, got {summary.get('mode')}")
    if int(summary.get("seed")) != job.seed:
        raise ValueError("Campaign summary reports a different seed")
    if int(summary.get("rounds_aborted") or 0) > 0:
        raise ValueError("Campaign aborted a round before completion")
    round_metrics = summary.get("round_metrics") or []
    if len(round_metrics) != job.rounds:
        raise ValueError(
            f"Expected {job.rounds} rounds, got {len(round_metrics)}"
        )
    expected_queries = job.rounds * job.budget
    if int(summary["queries_used"]) != expected_queries:
        raise ValueError(
            f"Expected queries_used={expected_queries} "
            f"(rounds {job.rounds} x budget {job.budget}), "
            f"got {summary['queries_used']}"
        )
    fold = summary.get("data_source", {}).get("fold_index")
    if fold != job.fold_index:
        raise ValueError("Campaign summary reports a different fold")
    final_metrics = summary.get("final_prediction_metrics") or {}
    expected_final = set(hierarchical_evaluation_contract()["metrics"]) | {"n"}
    missing_final = sorted(expected_final.difference(final_metrics))
    if missing_final:
        raise ValueError(
            f"final_prediction_metrics missing hierarchical keys: {missing_final}"
        )
    summary = dict(summary)
    summary["condition"] = job.mode
    return summary


def _run_one_job(
    job: CampaignJob,
    *,
    project_dir: Path,
    output_dir: Path,
    timeout_seconds: float | None,
    echo_progress: bool,
) -> CampaignJobResult:
    prefix = _job_id(job)
    stdout_path = output_dir / f"{prefix}.stdout.log"
    stderr_path = output_dir / f"{prefix}.stderr.log"
    print(
        f"{prefix} started rounds={job.rounds} budget={job.budget} "
        f"stderr={stderr_path}",
        flush=True,
    )
    process = subprocess.Popen(
        list(job.command),
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_box: list[str] = []
    stderr_box: list[str] = []

    def _capture_stdout() -> None:
        if process.stdout is None:
            return
        stdout_box.append(_drain_stream(process.stdout, stdout_path, None))

    def _capture_stderr() -> None:
        if process.stderr is None:
            return
        stdout_echo = sys.stderr if echo_progress else None
        stderr_box.append(_drain_stream(process.stderr, stderr_path, stdout_echo))

    stdout_thread = threading.Thread(target=_capture_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_capture_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait()
    stdout_thread.join()
    stderr_thread.join()
    stdout = stdout_box[0] if stdout_box else stdout_path.read_text(encoding="utf-8")
    if timed_out:
        return CampaignJobResult(
            job=job,
            status="timeout",
            returncode=124,
            stdout_log=str(stdout_path),
            stderr_log=str(stderr_path),
            summary=None,
            error=f"Campaign exceeded timeout of {timeout_seconds} seconds",
        )
    summary = None
    error = None
    status = "failed"
    returncode = int(process.returncode or 0)
    if returncode == 0:
        try:
            summary = _validate_summary(json.loads(stdout), job)
            status = "completed"
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as parse_error:
            error = f"Campaign output is not a valid summary: {parse_error}"
    else:
        error = f"Campaign exited with code {returncode}"
    return CampaignJobResult(
        job=job,
        status=status,
        returncode=returncode,
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
        summary=summary,
        error=error,
    )


def _build_jobs(args: argparse.Namespace) -> list[CampaignJob]:
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown = set(modes).difference(CONFIGS)
    if unknown:
        raise SystemExit(f"Unknown modes: {sorted(unknown)}")
    seeds = _parse_csv_ints(args.seeds)
    folds = _parse_csv_ints(args.folds)
    jobs: list[CampaignJob] = []
    for seed in seeds:
        for fold in folds:
            for mode in modes:
                _load_config(
                    mode, seed=seed, fold=fold, rounds=args.rounds, budget=args.budget
                )
                command = _job_command(
                    python_executable=sys.executable,
                    mode=mode,
                    seed=seed,
                    fold=fold,
                    rounds=args.rounds,
                    budget=args.budget,
                    quiet=args.quiet,
                    log_level=args.log_level,
                )
                jobs.append(
                    CampaignJob(
                        index=len(jobs),
                        mode=mode,
                        seed=seed,
                        fold_index=fold,
                        rounds=args.rounds,
                        budget=args.budget,
                        command=command,
                    )
                )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun random and fitness_direct on GB1 AL96. "
            "Each round selects --budget variants for wet validation (default 16). "
            "Final-test correlation metrics use the same Kermut/ESM-2 model as "
            "run_hierarchical_scientist.py; inference is limited to the selected "
            "batch (and a bounded fitness_direct screen), not the remaining library."
        )
    )
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--budget",
        type=int,
        default=16,
        help="Variants selected and wet-validated per round",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Concurrent campaign processes. Keep 1 unless CPU/RAM headroom is known.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-echo-progress",
        action="store_true",
        help="Write campaign stderr only to job_logs, not the parent terminal",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_from_args(args)
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")
    if args.budget < 1:
        raise SystemExit("--budget must be at least 1")
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be at least 1")
    jobs = _build_jobs(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        project_root() / "artifacts" / f"random-fitness-direct-b{args.budget}-{stamp}"
    )
    schedule = {
        "modes": [item.strip() for item in args.modes.split(",") if item.strip()],
        "seeds": _parse_csv_ints(args.seeds),
        "folds": _parse_csv_ints(args.folds),
        "rounds": args.rounds,
        "budget_per_round": args.budget,
        "queries_per_job": args.rounds * args.budget,
        "max_parallel": args.max_parallel,
        "hierarchical_config": HIERARCHICAL_CONFIG,
        "evaluation": hierarchical_evaluation_contract(),
        "fitness_model": KERMUT_MODEL,
        "candidate_limit": hierarchical_evaluation_contract()["candidate_limit"],
        "jobs": [
            {
                "mode": job.mode,
                "seed": job.seed,
                "fold_index": job.fold_index,
                "rounds": job.rounds,
                "budget_per_round": job.budget,
                "command": list(job.command),
            }
            for job in jobs
        ],
    }
    if args.dry_run:
        print(json.dumps({"output_dir": str(output_dir), **schedule}, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "schedule.json").write_text(
        json.dumps(schedule, indent=2), encoding="utf-8"
    )
    job_log_dir = output_dir / "job_logs"
    job_log_dir.mkdir()
    timeout = args.timeout_seconds if args.timeout_seconds > 0 else None
    echo_progress = not args.no_echo_progress and args.max_parallel == 1
    results: list[CampaignJobResult] = []
    workers = min(args.max_parallel, len(jobs))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="b16") as pool:
        future_map = {
            pool.submit(
                _run_one_job,
                job,
                project_dir=project_root(),
                output_dir=job_log_dir,
                timeout_seconds=timeout,
                echo_progress=echo_progress,
            ): job
            for job in jobs
        }
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(
                f"{_job_id(result.job)} status={result.status} "
                f"returncode={result.returncode}",
                flush=True,
            )
    results.sort(key=lambda item: item.job.index)
    summaries = [item.summary for item in results if item.summary is not None]
    aggregate = aggregate_runs(summaries, output_dir / "aggregate") if summaries else {}
    report = {
        "completed": sum(item.status == "completed" for item in results),
        "failed": sum(item.status != "completed" for item in results),
        "rounds": args.rounds,
        "budget_per_round": args.budget,
        "queries_per_job": args.rounds * args.budget,
        "aggregate": {key: str(value) for key, value in aggregate.items()},
        "results": [
            {
                "status": item.status,
                "job": {
                    "mode": item.job.mode,
                    "seed": item.job.seed,
                    "fold_index": item.job.fold_index,
                    "rounds": item.job.rounds,
                    "budget_per_round": item.job.budget,
                    "command": list(item.job.command),
                },
                "stdout_log": item.stdout_log,
                "stderr_log": item.stderr_log,
                "returncode": item.returncode,
                "error": item.error,
                "summary": item.summary,
            }
            for item in results
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "results"},
            indent=2,
        )
    )
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
