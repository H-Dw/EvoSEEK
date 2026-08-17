#!/usr/bin/env python3
"""Run native-client campaigns against random and fitness-direct baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.loop import run_campaign
from fitness_agents.reporting import aggregate_runs
from fitness_agents.utils.progress import add_logging_arguments, configure_from_args

PRESETS: dict[str, dict[str, str]] = {
    "al96": {
        "random": "configs/experiments/random_al96.yaml",
        "fitness_direct": "configs/experiments/fitness_direct_al96.yaml",
        "llm_agent": "configs/experiments/llm_agent_al96.yaml",
        "knowledge_agent": "configs/experiments/knowledge_agent_al96.yaml",
    },
    "demo": {
        "random": "configs/experiments/random.yaml",
        "fitness_direct": "configs/experiments/fitness_direct.yaml",
        "llm_agent": "configs/experiments/llm_agent.yaml",
        "knowledge_agent": "configs/experiments/knowledge_agent.yaml",
    },
}
DEFAULT_MODES = "random,fitness_direct,knowledge_agent"


def _parse_csv_ints(value: str | None) -> list[int | None]:
    if value is None or value.strip().lower() in {"", "config"}:
        return [None]
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parsed) != len(set(parsed)) or any(item < 0 for item in parsed):
        raise ValueError("Values must be unique non-negative integers")
    return list(parsed)


def _load_job(
    *, preset: str, mode: str, seed: int | None, fold: int | None,
    rounds: int | None, budget: int | None,
) -> Any:
    overrides: dict[str, Any] = {"run_label": f"{preset}-native-client-baseline"}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare native-client agents and deterministic baselines on paired folds"
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="al96")
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--folds", default="config")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    add_logging_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_from_args(args)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown = set(modes).difference(PRESETS[args.preset])
    if unknown:
        raise SystemExit(f"Unknown modes: {sorted(unknown)}")
    seeds = _parse_csv_ints(args.seeds)
    folds = _parse_csv_ints(args.folds)
    jobs = []
    for seed in seeds:
        for fold in folds:
            for mode in modes:
                config = _load_job(
                    preset=args.preset, mode=mode, seed=seed, fold=fold,
                    rounds=args.rounds, budget=args.budget,
                )
                jobs.append({
                    "mode": mode, "seed": config.seed,
                    "fold_index": config.task.fold_index,
                    "provider": config.llm.provider, "config": config,
                })
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or project_root() / "artifacts" / f"agent-baselines-{stamp}"
    schedule = [{key: value for key, value in job.items() if key != "config"} for job in jobs]
    if args.dry_run:
        print(json.dumps({"output_dir": str(output_dir), "jobs": schedule}, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "schedule.json").write_text(
        json.dumps(schedule, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summaries, results = [], []
    for job in jobs:
        public_job = {key: value for key, value in job.items() if key != "config"}
        try:
            summary = run_campaign(job["config"])
            summaries.append(summary)
            results.append({"status": "completed", "job": public_job, "summary": summary})
        except Exception as error:  # noqa: BLE001 - paired jobs continue independently
            results.append({
                "status": "failed", "job": public_job,
                "error": f"{type(error).__name__}: {error}",
            })
    aggregate = aggregate_runs(summaries, output_dir / "aggregate") if summaries else {}
    report = {
        "completed": sum(item["status"] == "completed" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
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
