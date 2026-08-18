from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from fitness_agents.config import (
    LocalKnowledgeRetrievalConfig,
    load_embedding_api_config,
    load_experiment_config,
    load_reranker_api_config,
)
from fitness_agents.local_knowledge import LocalKnowledgeBase
from fitness_agents.local_knowledge.api_backends import (
    build_embedding_backend,
    build_reranker_backend,
)
from fitness_agents.protein_features import ProteinTaskContext


def _vector_summary(vector: np.ndarray) -> dict[str, Any]:
    values = np.asarray(vector, dtype=np.float32)
    return {
        "dimension": int(values.size),
        "l2_norm": float(np.linalg.norm(values)),
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "preview": [float(value) for value in values[:8]],
    }


def _embedded_input_summary(
    text: str,
    vector: np.ndarray,
    *,
    include_text: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "input_characters": len(text),
        "vector": _vector_summary(vector),
    }
    if include_text:
        payload["text"] = text
    return payload


def _emit(payload: MappingLike, output: Path | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


MappingLike = dict[str, Any]


def _assert_english(label: str, text: str) -> None:
    if re.search(r"[\u3400-\u9fff]", text):
        raise ValueError(f"{label} must be English for this corpus profile")


def _api_retrieval_config(
    embedding_path: Path,
    reranker_path: Path | None = None,
) -> LocalKnowledgeRetrievalConfig:
    embedding = load_embedding_api_config(embedding_path)
    reranker = load_reranker_api_config(reranker_path) if reranker_path else None
    return LocalKnowledgeRetrievalConfig(
        mode="dense",
        dense_enabled=True,
        embedding_backend="api",
        embedding_api_config=embedding,
        reranker_backend="api" if reranker is not None else "none",
        reranker_api_config=reranker,
        strict_query_language=True,
    )


def _probe(args: argparse.Namespace) -> None:
    documents = list(args.document or ())
    for path in args.document_file or ():
        documents.append(path.read_text(encoding="utf-8"))
    prompts = list(args.prompt or ())
    if not prompts and not documents:
        raise ValueError("probe requires at least one --prompt, --document, or --document-file")
    for index, prompt in enumerate(prompts):
        _assert_english(f"prompt {index}", prompt)
    for index, document in enumerate(documents):
        _assert_english(f"document {index}", document)

    retrieval = _api_retrieval_config(args.embedding_config, args.reranker_config)
    embedding = build_embedding_backend(retrieval)
    if embedding is None:
        raise RuntimeError("API embedding backend was not created")
    payload: dict[str, Any] = {
        "embedding_backend": embedding.name,
        "embedding_fingerprint": embedding.fingerprint,
        "prompts": [],
        "documents": [],
    }
    if prompts:
        vectors = embedding.encode_queries(prompts)
        payload["prompts"] = [
            _embedded_input_summary(text, vector, include_text=args.include_text)
            for text, vector in zip(prompts, vectors, strict=True)
        ]
    if documents:
        vectors = embedding.encode_documents(documents)
        payload["documents"] = [
            _embedded_input_summary(text, vector, include_text=args.include_text)
            for text, vector in zip(documents, vectors, strict=True)
        ]

    reranker = build_reranker_backend(retrieval)
    if reranker is not None:
        if len(prompts) != 1 or not documents:
            raise ValueError(
                "reranker probe requires exactly one --prompt and one or more documents"
            )
        scores = reranker.score(prompts[0], documents)
        payload["reranker"] = {
            "backend": reranker.name,
            "fingerprint": getattr(reranker, "fingerprint", {}),
            "score_kind": reranker.score_kind,
            "ranking": [
                {"document_index": int(index), "score": float(scores[index])}
                for index in np.argsort(-scores)
            ],
        }
    _emit(payload, args.output)


def _index(args: argparse.Namespace) -> None:
    experiment = load_experiment_config(args.experiment_config)
    local = experiment.knowledge.local_knowledge
    if not local.enabled:
        raise ValueError("The experiment must enable local knowledge")
    embedding = load_embedding_api_config(args.embedding_config)
    reranker = load_reranker_api_config(args.reranker_config) if args.reranker_config else None
    retrieval = replace(
        local.retrieval,
        mode="hybrid" if local.retrieval.mode == "lexical" else local.retrieval.mode,
        dense_enabled=True,
        embedding_backend="api",
        embedding_model_path=None,
        embedding_api_config=embedding,
        reranker_backend="api" if reranker is not None else "none",
        reranker_model_path=None,
        reranker_api_config=reranker,
    )
    local = replace(local, retrieval=retrieval)
    task_context = ProteinTaskContext.from_task(experiment.task)
    knowledge = LocalKnowledgeBase(
        local,
        index_path=args.index_path or local.corpus_index_path or local.index_path,
        overlay_path=local.retrieval_overlay_path,
        protein_id=experiment.task.protein_id,
        protein_name=experiment.task.protein_name,
        protein_aliases=experiment.task.protein_aliases,
        protein_accessions=experiment.task.protein_accessions,
        reference_sequence=task_context.full_sequence,
    )
    try:
        report = knowledge.refresh()
        payload = {
            "build": asdict(report),
            "corpus": knowledge.index.stats(),
            "embedding_backend": knowledge.embedding_backend.name,
            "embedding_fingerprint": knowledge.embedding_backend.fingerprint,
            "reranker_backend": (
                knowledge.reranker_backend.name if knowledge.reranker_backend is not None else None
            ),
        }
    finally:
        knowledge.close()
    _emit(payload, args.output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire English prompt/corpus embeddings through a configured API"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="Embed prompts/documents and optionally rerank")
    probe.add_argument("--embedding-config", type=Path, required=True)
    probe.add_argument("--reranker-config", type=Path)
    probe.add_argument("--prompt", action="append")
    probe.add_argument("--document", action="append")
    probe.add_argument("--document-file", action="append", type=Path)
    probe.add_argument(
        "--include-text",
        action="store_true",
        help="Include raw prompt/document text in the JSON output (off by default)",
    )
    probe.add_argument("--output", type=Path)
    probe.set_defaults(handler=_probe)

    index = subparsers.add_parser("index", help="Build the configured corpus with API vectors")
    index.add_argument("--experiment-config", type=Path, required=True)
    index.add_argument("--embedding-config", type=Path, required=True)
    index.add_argument("--reranker-config", type=Path)
    index.add_argument("--index-path", type=Path)
    index.add_argument("--output", type=Path)
    index.set_defaults(handler=_index)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
