from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from fitness_agents.config import load_experiment_config
from fitness_agents.local_knowledge import LocalKnowledgeBase
from fitness_agents.local_knowledge.index import SQLiteLocalKnowledgeIndex
from fitness_agents.loop import run_campaign
from fitness_agents.protein_features import ProteinTaskContext
from fitness_agents.utils.progress import add_logging_arguments, configure_from_args


def _knowledge_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Manage an offline local knowledge index")
    parser.add_argument("action", choices=("index", "inspect"))
    parser.add_argument("config", help="Experiment YAML path containing local_knowledge config")
    parser.add_argument("--index-path", type=Path)
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config)
    local_config = config.knowledge.local_knowledge
    if not local_config.enabled:
        raise ValueError("knowledge local_knowledge.enabled must be true")
    index_path = (
        args.index_path.resolve()
        if args.index_path is not None
        else local_config.index_path
    )
    if index_path is None:
        index_path = (config.output_root / "local_knowledge" / f"{config.task.task_id}.sqlite").resolve()
    if args.action == "inspect":
        if not index_path.is_file():
            raise FileNotFoundError(f"Local knowledge index does not exist: {index_path}")
        index = SQLiteLocalKnowledgeIndex(index_path)
        try:
            print(json.dumps(index.stats(), ensure_ascii=False, indent=2))
        finally:
            index.close()
        return
    task_context = ProteinTaskContext.from_task(config.task)
    knowledge = LocalKnowledgeBase(
        local_config,
        index_path=index_path,
        protein_id=config.task.protein_id,
        protein_name=config.task.protein_name,
        protein_aliases=config.task.protein_aliases,
        protein_accessions=config.task.protein_accessions,
        reference_sequence=task_context.full_sequence,
    )
    try:
        report = knowledge.refresh()
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    finally:
        knowledge.close()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "knowledge":
        _knowledge_command(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description="Low-level fitness-agents campaign entry point")
    parser.add_argument("config", help="Experiment YAML path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--output-artifacts",
        help="Comma-separated subset of json,csv,markdown,svg,reasoning",
    )
    parser.add_argument("--output-top-k", type=int)
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)
    overrides = {"seed": args.seed} if args.seed is not None else None
    config = load_experiment_config(args.config, overrides=overrides)
    if args.fold_index is not None:
        config = replace(config, task=replace(config.task, fold_index=args.fold_index))
    if args.output_root is not None:
        config = replace(config, output_root=args.output_root.resolve())
    output = config.output
    if args.output_artifacts is not None:
        output = replace(
            output,
            artifacts=tuple(item.strip() for item in args.output_artifacts.split(",") if item.strip()),
        )
    if args.output_top_k is not None:
        output = replace(output, top_k=args.output_top_k)
    config = replace(config, output=output)
    print(json.dumps(run_campaign(config), indent=2))


if __name__ == "__main__":
    main()
