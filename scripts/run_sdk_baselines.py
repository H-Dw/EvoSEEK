#!/usr/bin/env python3
"""Run Agents SDK campaigns against random and fitness-direct baselines.

CampaignRunner still owns splits, oracle reveal, and approval. Agent modes use
`llm.runtime=agents_sdk`; random and fitness_direct never call an LLM.
"""
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
        "llm_agent": "configs/experiments/llm_agent_al96_sdk.yaml",
        "knowledge_agent": "configs/experiments/knowledge_agent_al96_sdk.yaml",
    },
    "demo": {
        "random": "configs/experiments/random.yaml",
        "fitness_direct": "configs/experiments/fitness_direct.yaml",
        "llm_agent": "configs/experiments/llm_agent.yaml",
        "knowledge_agent": "configs/experiments/knowledge_agent.yaml",
    },
}
AGENT_MODES = frozenset({"llm_agent", "knowledge_agent"})
DEFAULT_MODES = "random,fitness_direct,knowledge_agent"


def _parse_folds(value: str) -> list[int] | None:
    if value.strip().lower() in {"", "config"}:
        return None
    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(folds) != len(set(folds)):
        raise ValueError("Fold selection contains duplicates")
    if any(fold < 0 for fold in folds):
        raise ValueError("Fold indices must be non-negative")
    return folds


def _load_job_config(
    *,
    preset: str,
    mode: str,
    seed: int | None,
    rounds: int | None,
    budget: int | None,
    fold_index: int | None,
    task_config: str | None,
    model_config: str | None,
) -> Any:
    root = project_root()
    overrides: dict[str, Any] = {"run_label": f"{preset}-sdk-baseline"}
    if seed is not None:
        overrides["seed"] = seed
    if rounds is not None:
        overrides["rounds"] = rounds
    if budget is not None:
        overrides["budget_per_round"] = budget
    if task_config is not None:
        overrides["task_config"] = task_config
    if model_config is not None:
        overrides["model_config"] = model_config
    if mode in AGENT_MODES:
        overrides["llm"] = {"runtime": "agents_sdk"}
    config = load_experiment_config(root / PRESETS[preset][mode], overrides=overrides)
    if fold_index is not None:
        config = replace(config, task=replace(config.task, fold_index=fold_index))
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Agents SDK knowledge/LLM agents with random selection and "
            "fitness-predictor-only recommendation on the same split/seed/budget"
        )
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="al96")
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds; default is each YAML seed")
    parser.add_argument("--folds", default="config", help="config, or comma-separated fold indices")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    add_logging_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_from_args(args)
    root = project_root()
    modes = [value for value in args.modes.split(",") if value]
    unknown_modes = set(modes).difference(PRESETS[args.preset])
    if unknown_modes:
        raise SystemExit(f"Unknown modes for preset {args.preset!r}: {sorted(unknown_modes)}")
    seeds = (
        [int(value) for value in args.seeds.split(",") if value]
        if args.seeds
        else [None]
    )
    folds = _parse_folds(args.folds)
    if folds is None:
        fold_values: list[int | None] = [None]
    else:
        fold_values = list(folds)

    jobs: list[dict[str, Any]] = []
    for seed in seeds:
        for fold_index in fold_values:
            for mode in modes:
                config = _load_job_config(
                    preset=args.preset,
                    mode=mode,
                    seed=seed,
                    rounds=args.rounds,
                    budget=args.budget,
                    fold_index=fold_index,
                    task_config=args.task_config,
                    model_config=args.model_config,
                )
                jobs.append(
                    {
                        "mode": mode,
                        "seed": config.seed,
                        "fold_index": config.task.fold_index,
                        "runtime": config.llm.runtime if mode in AGENT_MODES else "none",
                        "provider": config.llm.provider,
                        "queries": config.rounds * config.budget_per_round,
                        "split_root": str(config.task.split_root) if config.task.split_root else None,
                        "config": config,
                    }
                )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or root / "artifacts" / f"sdk-baseline-comparison-{timestamp}"
    schedule = [
        {key: value for key, value in job.items() if key != "config"}
        for job in jobs
    ]
    if args.dry_run:
        print(json.dumps({"output_dir": str(output_dir), "jobs": schedule}, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "schedule.json").write_text(
        json.dumps({"preset": args.preset, "jobs": schedule}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summaries = []
    results = []
    for job in jobs:
        config = job["config"]
        print(
            f"mode={job['mode']} seed={job['seed']} fold={job['fold_index']} "
            f"runtime={job['runtime']}",
            flush=True,
        )
        try:
            summary = run_campaign(config)
            summaries.append(summary)
            results.append(
                {
                    "status": "completed",
                    "job": {key: value for key, value in job.items() if key != "config"},
                    "summary": summary,
                }
            )
        except Exception as error:  # noqa: BLE001 - keep remaining paired baselines
            results.append(
                {
                    "status": "failed",
                    "job": {key: value for key, value in job.items() if key != "config"},
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"failed: {type(error).__name__}: {error}", flush=True)

    aggregate_paths = (
        aggregate_runs(summaries, output_dir / "aggregate") if summaries else {}
    )
    report = {
        "completed": sum(item["status"] == "completed" for item in results),
        "failed": sum(item["status"] != "completed" for item in results),
        "aggregate": {key: str(value) for key, value in aggregate_paths.items()},
        "results": results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
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
