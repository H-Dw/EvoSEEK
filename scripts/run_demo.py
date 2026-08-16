#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.loop import run_campaign
from fitness_agents.utils.progress import add_logging_arguments, configure_from_args


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run one reproducible virtual evolution demo")
    parser.add_argument(
        "--config", type=Path, default=root / "configs/experiments/knowledge_agent.yaml"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--fold-index", type=int, default=None)
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)
    overrides = {key: value for key, value in {
        "seed": args.seed, "rounds": args.rounds, "budget_per_round": args.budget
    }.items() if value is not None}
    config = load_experiment_config(args.config, overrides=overrides)
    if args.fold_index is not None:
        config = replace(config, task=replace(config.task, fold_index=args.fold_index))
    summary = run_campaign(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
