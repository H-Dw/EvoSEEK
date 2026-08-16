#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.loop import run_campaign
from fitness_agents.reporting import aggregate_runs
from fitness_agents.utils.progress import add_logging_arguments, configure_from_args

CONFIGS = {
    "random": "configs/experiments/random.yaml",
    "fitness_direct": "configs/experiments/fitness_direct.yaml",
    "llm_agent": "configs/experiments/llm_agent.yaml",
    "knowledge_agent": "configs/experiments/knowledge_agent.yaml",
}


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run the four fair baseline modes")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--modes", default=",".join(CONFIGS))
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--model-config", default=None)
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    modes = [value for value in args.modes.split(",") if value]
    unknown = set(modes).difference(CONFIGS)
    if unknown:
        raise SystemExit(f"Unknown modes: {sorted(unknown)}")

    summaries = []
    for seed in seeds:
        for mode in modes:
            overrides = {"seed": seed, "run_label": "baseline"}
            if args.rounds is not None:
                overrides["rounds"] = args.rounds
            if args.budget is not None:
                overrides["budget_per_round"] = args.budget
            if args.task_config is not None:
                overrides["task_config"] = args.task_config
            if args.model_config is not None:
                overrides["model_config"] = args.model_config
            config = load_experiment_config(root / CONFIGS[mode], overrides=overrides)
            summaries.append(run_campaign(config))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = aggregate_runs(summaries, root / "artifacts" / f"baseline-comparison-{timestamp}")
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
