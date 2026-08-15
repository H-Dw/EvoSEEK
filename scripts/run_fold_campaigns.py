#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.reporting import aggregate_runs


@dataclass(frozen=True)
class FoldJob:
    fold_index: int
    seed: int
    command: tuple[str, ...]


@dataclass(frozen=True)
class FoldJobResult:
    fold_index: int
    seed: int
    status: str
    returncode: int
    stdout_log: str
    stderr_log: str
    summary: dict[str, Any] | None
    error: str | None = None


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


def _build_jobs(
    *,
    config_path: Path,
    folds: list[int],
    seed: int,
    python_executable: str,
) -> list[FoldJob]:
    return [
        FoldJob(
            fold_index=fold,
            seed=seed,
            command=(
                python_executable,
                "-m",
                "fitness_agents.cli",
                str(config_path),
                "--fold-index",
                str(fold),
                "--seed",
                str(seed),
            ),
        )
        for fold in folds
    ]


def _run_one_job(
    job: FoldJob,
    *,
    project_dir: Path,
    output_dir: Path,
    timeout_seconds: float | None,
) -> FoldJobResult:
    prefix = f"fold_{job.fold_index:02d}-seed_{job.seed}"
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
                summary = json.loads(completed.stdout)
                required = {
                    "run_id",
                    "mode",
                    "seed",
                    "round_metrics",
                    "final_prediction_metrics",
                    "queries_used",
                    "data_source",
                }
                if missing := required.difference(summary):
                    raise ValueError(f"Campaign summary is missing keys: {sorted(missing)}")
                if not summary["round_metrics"]:
                    raise ValueError("Campaign summary has no round metrics")
                if int(summary["data_source"]["fold_index"]) != job.fold_index:
                    raise ValueError("Campaign summary reports a different fold")
                status = "completed"
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as parse_error:
                error = f"Campaign output is not a valid fold summary: {parse_error}"
        else:
            error = f"Campaign exited with code {completed.returncode}"
        return FoldJobResult(
            fold_index=job.fold_index,
            seed=job.seed,
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
        return FoldJobResult(
            fold_index=job.fold_index,
            seed=job.seed,
            status="timeout",
            returncode=124,
            stdout_log=str(stdout_path),
            stderr_log=str(stderr_path),
            summary=None,
            error=f"Campaign exceeded timeout of {timeout_seconds} seconds",
        )


def run_fold_jobs(
    jobs: list[FoldJob],
    *,
    max_parallel: int,
    project_dir: Path,
    output_dir: Path,
    timeout_seconds: float | None = None,
) -> list[FoldJobResult]:
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=False)
    results: list[FoldJobResult] = []
    workers = min(max_parallel, len(jobs))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fold-campaign") as pool:
        future_by_fold = {
            pool.submit(
                _run_one_job,
                job,
                project_dir=project_dir,
                output_dir=output_dir,
                timeout_seconds=timeout_seconds,
            ): job.fold_index
            for job in jobs
        }
        for future in as_completed(future_by_fold):
            result = future.result()
            results.append(result)
            print(
                f"fold={result.fold_index:02d} status={result.status} "
                f"returncode={result.returncode}",
                flush=True,
            )
    return sorted(results, key=lambda item: item.fold_index)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Run manifest-backed campaign agents across folds in isolated processes"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/knowledge_agent_al96.yaml",
    )
    parser.add_argument("--folds", default="all", help="all or comma-separated fold indices")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_experiment_config(config_path)
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be at least 1")
    if config.task.split_root is None:
        raise SystemExit("Fold scheduler requires a manifest-backed task with split_root")
    manifest_path = config.task.split_root / "manifest.public.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Split manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_folds = int(manifest["n_folds"])
    folds = _parse_folds(args.folds, n_folds)
    seed = config.seed if args.seed is None else args.seed
    jobs = _build_jobs(
        config_path=config_path.resolve(),
        folds=folds,
        seed=seed,
        python_executable=sys.executable,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or root / "artifacts" / f"fold-campaigns-{timestamp}"
    schedule = {
        "config": str(config_path.resolve()),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "strategy": manifest["strategy"],
        "protocol_version": manifest["protocol_version"],
        "folds": folds,
        "seed": seed,
        "max_parallel": args.max_parallel,
        "jobs": [asdict(job) for job in jobs],
    }
    if args.dry_run:
        print(json.dumps(schedule, indent=2, ensure_ascii=False))
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "schedule.json").write_text(
        json.dumps(schedule, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # run_fold_jobs creates its output directory, so use a dedicated log child here.
    log_dir = output_dir / "fold_logs"
    timeout = args.timeout_seconds if args.timeout_seconds > 0 else None
    results = run_fold_jobs(
        jobs,
        max_parallel=args.max_parallel,
        project_dir=root,
        output_dir=log_dir,
        timeout_seconds=timeout,
    )
    result_payload = [asdict(result) for result in results]
    (output_dir / "fold_results.json").write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    successful = [result.summary for result in results if result.summary is not None]
    aggregate_paths = aggregate_runs(successful, output_dir / "aggregate") if successful else {}
    report = {
        "completed": sum(result.status == "completed" for result in results),
        "failed": sum(result.status != "completed" for result in results),
        "aggregate": {key: str(value) for key, value in aggregate_paths.items()},
        "results": result_payload,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
