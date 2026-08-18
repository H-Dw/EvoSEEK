#!/usr/bin/env python3
"""Run DeepSeek LLM + Qwen embedding/rerank RAG campaigns from index to artifacts.

Profiles:
  test  cheap complete loop on the GB1 demo split (baseline dry validation)
  demo  three-round demo split with Kermut dry validation
  al96  formal GB1-AL96 fold campaign with Kermut dry validation
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from fitness_agents.agents.remote_llm import load_project_env, resolve_secret
from fitness_agents.config import load_experiment_config, project_root
from fitness_agents.local_knowledge import LocalKnowledgeBase
from fitness_agents.loop import run_campaign
from fitness_agents.protein_features import ProteinTaskContext
from fitness_agents.utils.progress import add_logging_arguments, configure_from_args

PROFILES = {
    "test": "configs/experiments/knowledge_agent_qwen_demo.yaml",
    "demo": "configs/experiments/knowledge_agent_qwen_unified_api.yaml",
    "al96": "configs/experiments/knowledge_agent_qwen_al96.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="test")
    parser.add_argument("--config", type=Path, help="Override the profile experiment YAML")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--fold-index", type=int)
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip the explicit corpus refresh; the campaign still refreshes if needed",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def _require_dashscope_key() -> None:
    load_project_env()
    if resolve_secret("env:DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY"):
        return
    raise RuntimeError(
        "DASHSCOPE_API_KEY is not set. Add it to the gitignored .env file and rerun."
    )


def _refresh_index(config) -> dict[str, object]:
    local = config.knowledge.local_knowledge
    if not local.enabled:
        raise RuntimeError("The selected experiment does not enable local knowledge")
    task_context = ProteinTaskContext.from_task(config.task)
    knowledge = LocalKnowledgeBase(
        local,
        index_path=local.corpus_index_path or local.index_path,
        overlay_path=local.retrieval_overlay_path,
        protein_id=config.task.protein_id,
        protein_name=config.task.protein_name,
        protein_aliases=config.task.protein_aliases,
        protein_accessions=config.task.protein_accessions,
        reference_sequence=task_context.full_sequence,
    )
    try:
        report = knowledge.refresh()
        payload = {
            "build": asdict(report),
            "corpus": knowledge.index.stats(),
            "embedding_backend": getattr(knowledge.embedding_backend, "name", None),
            "reranker_backend": getattr(knowledge.reranker_backend, "name", None),
        }
    finally:
        knowledge.close()
    return payload


def main() -> int:
    args = parse_args()
    configure_from_args(args)
    _require_dashscope_key()
    root = project_root()
    config_path = args.config or (root / PROFILES[args.profile])
    overrides = {
        key: value
        for key, value in {
            "seed": args.seed,
            "rounds": args.rounds,
            "budget_per_round": args.budget,
        }.items()
        if value is not None
    }
    config = load_experiment_config(config_path, overrides=overrides)
    if args.fold_index is not None:
        config = replace(config, task=replace(config.task, fold_index=args.fold_index))
    if not args.skip_index:
        index_payload = _refresh_index(config)
        print(json.dumps({"stage": "index", **index_payload}, indent=2, default=str))
    summary = run_campaign(config)
    print(json.dumps({"stage": "campaign", **summary}, indent=2, default=str))
    run_dir = Path(str(summary.get("run_dir", "")))
    highlights = {
        "run_id": summary.get("run_id"),
        "run_dir": str(run_dir),
        "summary": str(run_dir / "summary.json"),
        "state": str(run_dir / "state.json"),
        "reasoning": str(run_dir / "reasoning.md"),
        "validation_records": str(run_dir / "validation_records.json"),
        "structured_kg": str(run_dir / "structured_kg.sqlite"),
        "round_01_local_rag": str(run_dir / "round_01" / "local_rag_retrieval.json"),
        "round_01_kg_interaction": str(run_dir / "round_01" / "kg_interaction.json"),
        "round_01_top_k": str(run_dir / "round_01" / "top_k.json"),
        "round_01_selection": str(run_dir / "round_01" / "selection.csv"),
        "round_01_hypothesis_assessment": str(run_dir / "round_01" / "hypothesis_assessment.json"),
    }
    print(json.dumps({"stage": "artifacts", **highlights}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
