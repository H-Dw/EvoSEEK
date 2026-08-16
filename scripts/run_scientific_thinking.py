#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone

from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.evaluation import ScientificThinkingEvaluator
from fitness_agents.loop import run_campaign
from fitness_agents.reporting import write_science_markdown
from fitness_agents.utils.progress import add_logging_arguments, configure_from_args


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Run knowledge ablation, score shuffle, and evidence deletion tests"
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--task-config", default=None)
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)
    overrides = {
        "seed": args.seed,
        "rounds": args.rounds,
        "budget_per_round": args.budget,
    }
    if args.task_config is not None:
        overrides["task_config"] = args.task_config
    base = load_experiment_config(
        root / "configs/experiments/knowledge_agent.yaml",
        overrides=overrides,
    )
    conditions = {
        "reference": replace(base, run_label="science-reference"),
        "knowledge_ablation": replace(
            base, knowledge_enabled=False, run_label="science-knowledge-ablation"
        ),
        "score_shuffle": replace(base, score_shuffle=True, run_label="science-score-shuffle"),
        "evidence_deletion": replace(
            base, evidence_deletion=True, run_label="science-evidence-deletion"
        ),
    }
    summaries = {name: run_campaign(config) for name, config in conditions.items()}
    evaluator = ScientificThinkingEvaluator()
    report = evaluator.evaluate(
        reference_dir=summaries["reference"]["run_dir"],
        knowledge_ablation_dir=summaries["knowledge_ablation"]["run_dir"],
        score_shuffle_dir=summaries["score_shuffle"]["run_dir"],
        evidence_deletion_dir=summaries["evidence_deletion"]["run_dir"],
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = root / "artifacts" / f"scientific-thinking-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = write_science_markdown(report, output_dir / "report.md")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), **report}, indent=2))


if __name__ == "__main__":
    main()
