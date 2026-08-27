from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

from fitness_agents.config import load_experiment_config
from fitness_agents.local_knowledge import LocalKnowledgeBase
from fitness_agents.local_knowledge.api_backends import build_embedding_backend
from fitness_agents.local_knowledge.index import (
    SQLiteLocalKnowledgeIndex,
    preflight_local_knowledge,
)
from fitness_agents.local_knowledge.overlay import SQLiteRetrievalOverlay
from fitness_agents.loop import run_campaign
from fitness_agents.protein_features import ProteinTaskContext
from fitness_agents.utils.progress import add_logging_arguments, configure_from_args


def _knowledge_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Manage an offline local knowledge index")
    parser.add_argument("action", choices=("index", "rebuild", "preflight", "inspect"))
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
        else local_config.corpus_index_path or local_config.index_path
    )
    if index_path is None:
        index_path = (config.output_root / "local_knowledge" / f"{config.task.task_id}.sqlite").resolve()
    if args.action == "preflight":
        print(json.dumps(preflight_local_knowledge(local_config), ensure_ascii=False, indent=2))
        return
    if args.action == "inspect":
        if not index_path.is_file():
            raise FileNotFoundError(f"Local knowledge index does not exist: {index_path}")
        index = SQLiteLocalKnowledgeIndex(index_path, read_only=True)
        try:
            payload = {"corpus": index.stats()}
            overlay_path = local_config.retrieval_overlay_path
            if overlay_path is not None and overlay_path.is_file():
                overlay = SQLiteRetrievalOverlay(overlay_path)
                try:
                    payload["task_overlay"] = overlay.stats()
                finally:
                    overlay.close()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        finally:
            index.close()
        return
    if args.action == "rebuild":
        temporary = index_path.with_name(f"{index_path.name}.v6-building-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"Refusing to overwrite rebuild sidecar: {temporary}")
        backend = build_embedding_backend(local_config.retrieval)
        index = SQLiteLocalKnowledgeIndex(temporary)
        try:
            report = index.build(local_config, embedding_backend=backend)
        except Exception:
            index.close()
            if temporary.is_file():
                temporary.unlink()
            raise
        else:
            index.close()
            temporary.replace(index_path)
            print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
            return
    task_context = ProteinTaskContext.from_task(config.task)
    knowledge = LocalKnowledgeBase(
        local_config,
        index_path=index_path,
        overlay_path=local_config.retrieval_overlay_path,
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


def _design_command(argv: list[str]) -> None:
    from fitness_agents.interaction import EvolutionApplicationService

    parser = argparse.ArgumentParser(description="Preview or run an open sequence design")
    parser.add_argument("config", help="Trusted open-design experiment YAML")
    parser.add_argument("--prompt", required=True, help="Natural-language design objective")
    parser.add_argument("--fasta", type=Path, help="Optional reference FASTA/plain sequence")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Run after printing the validated preview; otherwise preview only",
    )
    args = parser.parse_args(argv)
    sequence_text = args.fasta.read_text(encoding="utf-8") if args.fasta else None
    service = EvolutionApplicationService(args.config)
    preview = service.preview(args.prompt, sequence_text=sequence_text)
    print(json.dumps(preview.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if args.confirm:
        result = service.run(preview.preview_id, confirmed=True)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


def _serve_command(argv: list[str]) -> None:
    from fitness_agents.interaction.gradio_app import launch_app

    parser = argparse.ArgumentParser(description="Launch the local open-design interface")
    parser.add_argument("config", help="Trusted open-design experiment YAML")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    launch_app(args.config, host=args.host, port=args.port)


def _apply_local_knowledge_paths(config, *, index_path: Path | None, overlay_path: Path | None):
    if index_path is None and overlay_path is None:
        return config
    local = config.knowledge.local_knowledge
    updates: dict[str, Path] = {}
    if index_path is not None:
        resolved = index_path.resolve()
        updates["index_path"] = resolved
        updates["corpus_index_path"] = resolved
    if overlay_path is not None:
        updates["retrieval_overlay_path"] = overlay_path.resolve()
    return replace(
        config,
        knowledge=replace(config.knowledge, local_knowledge=replace(local, **updates)),
    )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "deep-research":
        from fitness_agents.deep_research.cli import main as deep_research_main

        raise SystemExit(deep_research_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "knowledge":
        _knowledge_command(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "design":
        _design_command(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        _serve_command(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description="Low-level EvoSEEK campaign entry point")
    parser.add_argument("config", help="Experiment YAML path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--budget-per-round", type=int)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--run-label")
    parser.add_argument("--condition")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--output-artifacts",
        help="Comma-separated subset of json,csv,markdown,svg,reasoning",
    )
    parser.add_argument("--output-top-k", type=int)
    parser.add_argument(
        "--local-knowledge-index",
        type=Path,
        help="Override the local knowledge corpus sqlite path for this job",
    )
    parser.add_argument(
        "--local-knowledge-overlay",
        type=Path,
        help="Override the per-task retrieval overlay sqlite path for this job",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)
    overrides: dict[str, object] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.rounds is not None:
        overrides["rounds"] = args.rounds
    if args.budget_per_round is not None:
        overrides["budget_per_round"] = args.budget_per_round
    if args.candidate_limit is not None:
        overrides["candidate_limit"] = args.candidate_limit
    if args.run_label is not None:
        overrides["run_label"] = args.run_label
    if args.condition is not None:
        overrides["condition"] = args.condition
    config = load_experiment_config(args.config, overrides=overrides or None)
    if args.fold_index is not None:
        config = replace(config, task=replace(config.task, fold_index=args.fold_index))
    if args.output_root is not None:
        config = replace(config, output_root=args.output_root.resolve())
    config = _apply_local_knowledge_paths(
        config,
        index_path=args.local_knowledge_index,
        overlay_path=args.local_knowledge_overlay,
    )
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
