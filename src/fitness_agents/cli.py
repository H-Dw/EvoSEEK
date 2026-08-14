from __future__ import annotations

import argparse
import json

from fitness_agents.config import load_experiment_config
from fitness_agents.loop import run_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-level fitness-agents campaign entry point")
    parser.add_argument("config", help="Experiment YAML path")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    overrides = {"seed": args.seed} if args.seed is not None else None
    print(json.dumps(run_campaign(load_experiment_config(args.config, overrides=overrides)), indent=2))


if __name__ == "__main__":
    main()

