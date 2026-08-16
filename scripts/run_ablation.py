#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.loop import run_campaign
from fitness_agents.reporting import aggregate_runs
from fitness_agents.utils.progress import add_logging_arguments, configure_from_args

ABLATIONS = [
    "no_physchem",
    "no_conservation",
    "no_structure",
    "no_kg",
    "no_knowledge",
    "no_uq",
    "no_llm",
]


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run single-factor module ablations")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--ablations", default=",".join(ABLATIONS))
    parser.add_argument("--task-config", default=None)
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)
    names = [value for value in args.ablations.split(",") if value]
    unknown = set(names).difference(ABLATIONS)
    if unknown:
        raise SystemExit(f"Unknown ablations: {sorted(unknown)}")

    base_overrides = {
        "seed": args.seed,
        "rounds": args.rounds,
        "budget_per_round": args.budget,
        "acquisition": "ucb",
        "run_label": "ablation-reference",
    }
    if args.task_config is not None:
        base_overrides["task_config"] = args.task_config
    base_path = root / "configs/experiments/knowledge_agent.yaml"
    summaries = [run_campaign(load_experiment_config(base_path, overrides=base_overrides))]
    for name in names:
        overrides = {**base_overrides, "run_label": f"ablation-{name}"}
        config = load_experiment_config(
            base_path,
            overrides=overrides,
            ablation_path=root / f"configs/ablation/{name}.yaml",
        )
        summaries.append(run_campaign(config))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = aggregate_runs(summaries, root / "artifacts" / f"ablation-{timestamp}")
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
