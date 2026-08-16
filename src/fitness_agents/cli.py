from __future__ import annotations

import argparse
import json
from dataclasses import replace

from fitness_agents.config import load_experiment_config
from fitness_agents.loop import run_campaign
from fitness_agents.utils.progress import add_logging_arguments, configure_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-level fitness-agents campaign entry point")
    parser.add_argument("config", help="Experiment YAML path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold-index", type=int)
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)
    overrides = {"seed": args.seed} if args.seed is not None else None
    config = load_experiment_config(args.config, overrides=overrides)
    if args.fold_index is not None:
        config = replace(config, task=replace(config.task, fold_index=args.fold_index))
    print(json.dumps(run_campaign(config), indent=2))


if __name__ == "__main__":
    main()
