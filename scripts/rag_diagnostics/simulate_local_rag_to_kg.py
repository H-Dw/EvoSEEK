from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fitness_agents.config import load_experiment_config
from fitness_agents.kg_knowledge import (
    BuildContext,
    KnowledgeGraphBuilder,
    LocalRAGKnowledgeAdapter,
    SQLiteGraphSink,
)
from fitness_agents.local_knowledge import LocalKnowledgeBase
from fitness_agents.local_knowledge.index import SQLiteLocalKnowledgeIndex
from fitness_agents.plugin_registry import PluginRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a real local SentenceTransformer index, compare lexical/dense/hybrid "
            "retrieval, and materialize one retrieved result into the structured KG."
        )
    )
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/knowledge_agent.yaml",
        help="Experiment configuration whose local knowledge roots and policies are audited.",
    )
    parser.add_argument(
        "--embedding-model",
        required=True,
        help="Existing local SentenceTransformer directory. Runtime downloads are not allowed.",
    )
    parser.add_argument(
        "--gold-queries",
        default="configs/diagnostics/local_rag_gold_queries.yaml",
        help="YAML file containing expected knowledge types for diagnostic queries.",
    )
    parser.add_argument(
        "--output-dir",
        help="New output directory. Defaults to a timestamped artifacts/rag-diagnostics path.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--round-id", type=int, default=1)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--minimum-hit-at-3", type=float, default=0.75)
    parser.add_argument("--maximum-overbudget-ratio", type=float, default=0.05)
    parser.add_argument("--maximum-truncated-chunk-ratio", type=float, default=0.05)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_queries(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise TypeError("Gold query YAML must contain a queries list")
    output = []
    for raw in payload["queries"]:
        if not isinstance(raw, dict):
            raise TypeError("Each gold query must be a mapping")
        required = {"id", "query", "expected_knowledge_type", "expected_terms"}
        if not required.issubset(raw):
            raise ValueError(f"Gold query is missing fields: {sorted(required.difference(raw))}")
        output.append(dict(raw))
    return output


def ranking_rows(index: Any, ranking: tuple[tuple[str, float], ...]) -> list[dict[str, Any]]:
    records = index.get_chunks(tuple(chunk_id for chunk_id, _score in ranking))
    rows = []
    for rank, (chunk_id, score) in enumerate(ranking, start=1):
        record = records[chunk_id]
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "score": float(score),
                "knowledge_type": record["knowledge_type"],
                "artifact_uri": record["artifact_uri"],
                "section_path": list(record["section_path"]),
                "text_excerpt": record["text"][:320],
            }
        )
    return rows


def expected_rank(rows: list[dict[str, Any]], knowledge_type: str) -> int | None:
    for row in rows:
        if row["knowledge_type"] == knowledge_type:
            return int(row["rank"])
    return None


def embedding_audit(connection: sqlite3.Connection) -> dict[str, Any]:
    chunk_count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    rows = connection.execute(
        "SELECT dimension, vector FROM embeddings ORDER BY chunk_id"
    ).fetchall()
    norms = []
    finite = True
    for dimension, blob in rows:
        vector = np.frombuffer(blob, dtype=np.float32)
        if len(vector) != int(dimension) or not np.isfinite(vector).all():
            finite = False
        norms.append(float(np.linalg.norm(vector)))
    return {
        "chunk_count": chunk_count,
        "embedding_count": len(rows),
        "coverage": (len(rows) / chunk_count) if chunk_count else 0.0,
        "dimensions": sorted({int(row[0]) for row in rows}),
        "all_finite": finite,
        "norm_min": min(norms) if norms else None,
        "norm_max": max(norms) if norms else None,
        "maximum_unit_norm_deviation": (
            max(abs(item - 1.0) for item in norms) if norms else None
        ),
    }


def chunk_audit(connection: sqlite3.Connection, configured_tokens: int) -> dict[str, Any]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT chunk_id, knowledge_type, token_count, LENGTH(text) AS characters, "
        "section_path_json, artifact_uri, text FROM chunks ORDER BY token_count DESC"
    ).fetchall()
    chunks = [
        {
            "chunk_id": str(row["chunk_id"]),
            "knowledge_type": str(row["knowledge_type"]),
            "token_count": int(row["token_count"]),
            "characters": int(row["characters"]),
            "section_path": json.loads(row["section_path_json"]),
            "artifact_uri": str(row["artifact_uri"]),
            "text_start": str(row["text"])[:160],
        }
        for row in rows
    ]
    overbudget = [item for item in chunks if item["token_count"] > configured_tokens]
    return {
        "configured_chunk_tokens": configured_tokens,
        "chunk_count": len(chunks),
        "minimum_token_count": min((item["token_count"] for item in chunks), default=0),
        "maximum_token_count": max((item["token_count"] for item in chunks), default=0),
        "overbudget_count": len(overbudget),
        "overbudget_ratio": (len(overbudget) / len(chunks)) if chunks else 0.0,
        "overbudget_chunks": overbudget,
        "all_chunks": chunks,
    }


def model_input_audit(embedding_backend: Any, connection: sqlite3.Connection) -> dict[str, Any]:
    model = embedding_backend.model
    tokenizer = model.tokenizer
    maximum_tokens = int(model.max_seq_length)
    rows = connection.execute(
        "SELECT chunk_id, knowledge_type, text FROM chunks ORDER BY chunk_id"
    ).fetchall()
    details = []
    for row in rows:
        token_ids = tokenizer.encode(
            str(row["text"]),
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
        details.append(
            {
                "chunk_id": str(row["chunk_id"]),
                "knowledge_type": str(row["knowledge_type"]),
                "model_token_count": len(token_ids),
                "model_max_seq_length": maximum_tokens,
                "would_truncate": len(token_ids) > maximum_tokens,
                "retained_token_ratio_upper_bound": min(1.0, maximum_tokens / len(token_ids)),
            }
        )
    truncated = [item for item in details if item["would_truncate"]]
    return {
        "model_max_seq_length": maximum_tokens,
        "chunk_count": len(details),
        "truncated_chunk_count": len(truncated),
        "truncated_chunk_ratio": (len(truncated) / len(details)) if details else 0.0,
        "maximum_model_token_count": max(
            (item["model_token_count"] for item in details), default=0
        ),
        "minimum_retained_token_ratio_upper_bound": min(
            (item["retained_token_ratio_upper_bound"] for item in details), default=1.0
        ),
        "chunks": details,
    }


def entity_payload(entity: Any) -> dict[str, Any]:
    return {
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "layer": entity.layer.value,
        "modalities": sorted(item.value for item in entity.modalities),
        "properties": jsonable(entity.properties),
        "source_ids": list(entity.source_ids),
        "source_group": entity.source_group,
        "confidence": entity.confidence,
        "valid_from_round": entity.valid_from_round,
        "valid_to_round": entity.valid_to_round,
    }


def relation_payload(relation: Any) -> dict[str, Any]:
    return {
        "relation_id": relation.relation_id,
        "subject_id": relation.subject_id,
        "predicate": relation.predicate,
        "object_id": relation.object_id,
        "layer": relation.layer.value,
        "modalities": sorted(item.value for item in relation.modalities),
        "source_ids": list(relation.source_ids),
        "evidence_ids": list(relation.evidence_ids),
        "source_group": relation.source_group,
        "confidence": relation.confidence,
        "context_id": relation.context_id,
        "valid_from_round": relation.valid_from_round,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Local RAG dense retrieval → KG diagnostic",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Model: `{report['model']['path']}`",
        f"- Dimension: `{report['model']['dimension']}`",
        f"- Overall credible: `{report['verdict']['credible']}`",
        "",
        "## Chunk audit",
        "",
        f"- Chunks: `{report['chunk_audit']['chunk_count']}`",
        f"- Configured chunk tokens: `{report['chunk_audit']['configured_chunk_tokens']}`",
        f"- Maximum reported tokens: `{report['chunk_audit']['maximum_token_count']}`",
        f"- Over-budget ratio: `{report['chunk_audit']['overbudget_ratio']:.3f}`",
        "",
        "## Vector audit",
        "",
        f"- Coverage: `{report['embedding_audit']['coverage']:.3f}`",
        f"- All finite: `{report['embedding_audit']['all_finite']}`",
        f"- Dimensions: `{report['embedding_audit']['dimensions']}`",
        (
            f"- Maximum unit-norm deviation: "
            f"`{report['embedding_audit']['maximum_unit_norm_deviation']}`"
        ),
        (
            f"- Dense enablement backfills an existing lexical index: "
            f"`{report['incremental_dense_enablement']['backfill_supported']}`"
        ),
        f"- Model max sequence length: `{report['model_input_audit']['model_max_seq_length']}`",
        f"- Truncated chunk ratio: `{report['model_input_audit']['truncated_chunk_ratio']:.3f}`",
        "",
        "## Retrieval quality",
        "",
        "| Query | Expected type | Lexical rank | Dense rank | Hybrid rank |",
        "|---|---|---:|---:|---:|",
    ]
    for item in report["queries"]:
        lines.append(
            "| {id} | {expected} | {lexical} | {dense} | {hybrid} |".format(
                id=item["id"],
                expected=item["expected_knowledge_type"],
                lexical=item["expected_rank"]["lexical"] or "—",
                dense=item["expected_rank"]["dense"] or "—",
                hybrid=item["expected_rank"]["hybrid"] or "—",
            )
        )
    lines.extend(
        [
            "",
            f"- Lexical hit@3: `{report['metrics']['lexical_hit_at_3']:.3f}`",
            f"- Dense hit@3: `{report['metrics']['dense_hit_at_3']:.3f}`",
            f"- Hybrid hit@3: `{report['metrics']['hybrid_hit_at_3']:.3f}`",
            "",
            "## KG example",
            "",
            f"- Query: `{report['kg_example']['query']}`",
            f"- Retrieved chunk: `{report['kg_example']['retrieved_chunk']['chunk_id']}`",
            f"- Entity types: `{report['kg_example']['entity_types']}`",
            f"- Predicates: `{report['kg_example']['predicates']}`",
            "",
            (
                "See `diagnostic.json` for complete rankings, chunk excerpts, vectors metadata, "
                "KG entities, and relations."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_path = resolve_repo_path(args.embedding_model).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Embedding model directory does not exist: {model_path}")
    config_path = resolve_repo_path(args.experiment_config).resolve()
    gold_path = resolve_repo_path(args.gold_queries).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        resolve_repo_path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT / "artifacts" / "rag-diagnostics" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    experiment = load_experiment_config(config_path)
    original_local = experiment.knowledge.local_knowledge
    retrieval = replace(
        original_local.retrieval,
        mode="hybrid",
        dense_enabled=True,
        embedding_model_path=model_path,
        reranker_model_path=None,
    )
    local_config = replace(
        original_local,
        index_path=output_dir / "local_knowledge.sqlite",
        corpus_index_path=output_dir / "local_knowledge.sqlite",
        retrieval_overlay_path=output_dir / "retrieval_overlay.sqlite",
        retrieval=retrieval,
    )

    selection_enablement = {"guarded": False, "error": None}
    try:
        replace(local_config.kg_update, contributes_to_selection=True)
    except ValueError as error:
        selection_enablement = {"guarded": True, "error": str(error)}

    model_load_started = time.perf_counter()
    knowledge = LocalKnowledgeBase(
        local_config,
        index_path=local_config.index_path,
        protein_id=experiment.task.protein_id,
        protein_name=experiment.task.protein_name,
        protein_aliases=experiment.task.protein_aliases,
        protein_accessions=experiment.task.protein_accessions,
        reference_sequence=experiment.task.reference_sequence,
    )
    model_load_seconds = time.perf_counter() - model_load_started
    sink: SQLiteGraphSink | None = None
    try:
        build_started = time.perf_counter()
        build_report = knowledge.refresh()
        build_seconds = time.perf_counter() - build_started
        chunk_report = chunk_audit(
            knowledge.index.connection, local_config.ingestion.chunk_tokens
        )
        vector_report = embedding_audit(knowledge.index.connection)
        model_input_report = model_input_audit(
            knowledge.embedding_backend, knowledge.index.connection
        )
        upgrade_index = SQLiteLocalKnowledgeIndex(output_dir / "lexical-to-dense-upgrade.sqlite")
        try:
            upgrade_index.build(local_config, guard=knowledge.guard, embedding_backend=None)
            before_upgrade = upgrade_index.stats()
            upgrade_index.build(
                local_config,
                guard=knowledge.guard,
                embedding_backend=knowledge.embedding_backend,
            )
            after_upgrade = upgrade_index.stats()
            incremental_dense_enablement = {
                "chunks": int(after_upgrade["chunks"]),
                "embeddings_before_dense_enablement": int(before_upgrade["embeddings"]),
                "embeddings_after_dense_enablement": int(after_upgrade["embeddings"]),
                "backfill_supported": bool(
                    after_upgrade["chunks"] > 0
                    and after_upgrade["embeddings"] == after_upgrade["chunks"]
                ),
                "manifest_after_dense_enablement": after_upgrade["manifest_hash"],
            }
        finally:
            upgrade_index.close()
        queries = load_queries(gold_path)
        query_reports = []
        retrieval_results = []
        dense_query_seconds: list[float] = []
        hybrid_query_seconds: list[float] = []
        for case in queries:
            query = str(case["query"])
            expected_type = str(case["expected_knowledge_type"])
            lexical = ranking_rows(
                knowledge.index,
                knowledge.index.lexical_search(
                    query, limit=local_config.retrieval.lexical_candidates
                ),
            )
            dense_started = time.perf_counter()
            dense = ranking_rows(
                knowledge.index,
                knowledge.index.dense_search(
                    query,
                    limit=local_config.retrieval.dense_candidates,
                    embedding_backend=knowledge.embedding_backend,
                    minimum_similarity=local_config.retrieval.minimum_dense_similarity,
                    max_exact_chunks=local_config.retrieval.max_exact_dense_chunks,
                ),
            )
            dense_query_seconds.append(time.perf_counter() - dense_started)
            hybrid_started = time.perf_counter()
            result = knowledge.retrieve(
                query=query,
                intent="diagnostic",
                round_id=args.round_id,
                top_k=args.top_k,
            )
            hybrid_query_seconds.append(time.perf_counter() - hybrid_started)
            retrieval_results.append(result)
            hybrid = [
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "scores": chunk.scores,
                    "knowledge_type": chunk.knowledge_type,
                    "artifact_uri": chunk.artifact_uri,
                    "section_path": list(chunk.section_path),
                    "text_excerpt": chunk.text[:320],
                }
                for rank, chunk in enumerate(result.chunks, start=1)
            ]
            query_reports.append(
                {
                    "id": case["id"],
                    "query": query,
                    "expected_knowledge_type": expected_type,
                    "expected_terms": list(case["expected_terms"]),
                    "expected_rank": {
                        "lexical": expected_rank(lexical, expected_type),
                        "dense": expected_rank(dense, expected_type),
                        "hybrid": expected_rank(hybrid, expected_type),
                    },
                    "lexical": lexical[: args.top_k],
                    "dense": dense[: args.top_k],
                    "hybrid": hybrid,
                }
            )

        count = len(query_reports)
        def metric(channel: str, k: int) -> float:
            return sum(
                item["expected_rank"][channel] is not None
                and item["expected_rank"][channel] <= k
                for item in query_reports
            ) / count
        metrics = {
            "lexical_hit_at_1": metric("lexical", 1),
            "lexical_hit_at_3": metric("lexical", 3),
            "dense_hit_at_1": metric("dense", 1),
            "dense_hit_at_3": metric("dense", 3),
            "hybrid_hit_at_1": metric("hybrid", 1),
            "hybrid_hit_at_3": metric("hybrid", 3),
        }
        no_answer_result = knowledge.retrieve(
            query="municipal tax filing deadlines and urban parking permits",
            intent="diagnostic_no_answer",
            round_id=args.round_id,
            top_k=args.top_k,
        )

        example_result = retrieval_results[0]
        if not example_result.chunks:
            raise RuntimeError("The example query returned no chunks")
        expected_example_type = str(query_reports[0]["expected_knowledge_type"])
        example_chunk = next(
            (
                item
                for item in example_result.chunks
                if item.knowledge_type == expected_example_type
            ),
            example_result.chunks[0],
        )
        registry = PluginRegistry("knowledge_adapter")
        registry.register(
            "local_rag",
            LocalRAGKnowledgeAdapter(
                knowledge.guard,
                publication_catalog=knowledge.publication_catalog,
            ),
        )
        sink = SQLiteGraphSink(output_dir / "structured_kg.sqlite")
        built = KnowledgeGraphBuilder(registry, sinks=(sink,), strict=True).build(
            BuildContext(
                run_id="local-rag-diagnostic",
                round_id=args.round_id,
                protein_id=experiment.task.protein_id,
                resources={"local_retrieval_results": (example_result,)},
            )
        )
        example_claims = [
            claim
            for claim in example_result.claims
            if example_chunk.chunk_id in claim.evidence_chunk_ids
        ]
        linked_ids = {
            example_chunk.document_id,
            example_chunk.chunk_id,
            *(claim.claim_id for claim in example_claims),
            f"evidence:ev:local_rag:{example_chunk.chunk_id.split(':', 1)[-1]}",
        }
        linked_ids.update(
            item.entity_id
            for item in built.snapshot.entities
            if item.entity_type in {"CitationSupport", "Publication"}
        )
        entities = [
            entity_payload(item)
            for item in built.snapshot.entities
            if item.entity_id in linked_ids
        ]
        relations = [
            relation_payload(item)
            for item in built.snapshot.relations
            if item.subject_id in linked_ids or item.object_id in linked_ids
        ]
        kg_example = {
            "query": query_reports[0]["query"],
            "retrieved_chunk": {
                "chunk_id": example_chunk.chunk_id,
                "document_id": example_chunk.document_id,
                "knowledge_type": example_chunk.knowledge_type,
                "scores": example_chunk.scores,
                "artifact_uri": example_chunk.artifact_uri,
                "section_path": list(example_chunk.section_path),
                "text": example_chunk.text,
            },
            "claims": [jsonable(item.__dict__) for item in example_claims],
            "entities": entities,
            "relations": relations,
            "entity_types": sorted({item["entity_type"] for item in entities}),
            "predicates": sorted({item["predicate"] for item in relations}),
            "sqlite_path": str(sink.path),
        }

        vector_ok = bool(
            math.isclose(vector_report["coverage"], 1.0)
            and vector_report["all_finite"]
            and vector_report["maximum_unit_norm_deviation"] is not None
            and vector_report["maximum_unit_norm_deviation"] < 1e-3
        )
        retrieval_ok = bool(
            metrics["dense_hit_at_3"] >= args.minimum_hit_at_3
            and metrics["hybrid_hit_at_3"] >= args.minimum_hit_at_3
        )
        chunk_ok = bool(
            chunk_report["overbudget_ratio"] <= args.maximum_overbudget_ratio
        )
        model_input_ok = bool(
            model_input_report["truncated_chunk_ratio"]
            <= args.maximum_truncated_chunk_ratio
        )
        backfill_ok = bool(incremental_dense_enablement["backfill_supported"])
        no_answer_ok = bool(
            not no_answer_result.chunks
            and "no_answer_above_retrieval_threshold" in no_answer_result.warnings
        )
        verdict = {
            "vector_storage_and_cosine_valid": vector_ok,
            "retrieval_meets_gold_threshold": retrieval_ok,
            "chunking_meets_budget_threshold": chunk_ok,
            "model_input_preserves_chunks": model_input_ok,
            "incremental_dense_enablement_valid": backfill_ok,
            "irrelevant_query_returns_no_answer": no_answer_ok,
            "credible": (
                vector_ok
                and retrieval_ok
                and chunk_ok
                and model_input_ok
                and backfill_ok
                and no_answer_ok
            ),
            "thresholds": {
                "minimum_hit_at_3": args.minimum_hit_at_3,
                "maximum_overbudget_ratio": args.maximum_overbudget_ratio,
                "maximum_truncated_chunk_ratio": args.maximum_truncated_chunk_ratio,
            },
        }
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment_config": str(config_path),
            "gold_queries": str(gold_path),
            "default_configuration": {
                "corpus_index_path": str(original_local.corpus_index_path),
                "retrieval_overlay_path": str(original_local.retrieval_overlay_path),
                "retrieval_mode": original_local.retrieval.mode,
                "dense_enabled": original_local.retrieval.dense_enabled,
                "embedding_model_path": original_local.retrieval.embedding_model_path,
                "allow_remote_context": original_local.allow_remote_context,
                "contributes_to_selection": original_local.kg_update.contributes_to_selection,
            },
            "selection_enablement": selection_enablement,
            "model": {
                "path": str(model_path),
                "backend_name": knowledge.embedding_backend.name,
                "dimension": knowledge.embedding_backend.dimension,
                "fingerprint": knowledge.embedding_backend.fingerprint,
            },
            "index_build": jsonable(build_report.__dict__),
            "chunk_audit": chunk_report,
            "embedding_audit": vector_report,
            "model_input_audit": model_input_report,
            "incremental_dense_enablement": incremental_dense_enablement,
            "metrics": metrics,
            "cpu_timing": {
                "device": str(knowledge.embedding_backend.model.device),
                "model_load_seconds": model_load_seconds,
                "index_build_seconds": build_seconds,
                "dense_query_mean_milliseconds": (
                    1000.0 * sum(dense_query_seconds) / len(dense_query_seconds)
                ),
                "hybrid_query_mean_milliseconds": (
                    1000.0 * sum(hybrid_query_seconds) / len(hybrid_query_seconds)
                ),
                "corpus_chunks": len(model_input_report["chunks"]),
                "scope_warning": "Small-corpus timing; not a scale benchmark.",
            },
            "no_answer": jsonable(no_answer_result.__dict__),
            "queries": query_reports,
            "kg_example": kg_example,
            "verdict": verdict,
        }
        (output_dir / "diagnostic.json").write_text(
            json.dumps(jsonable(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_markdown(report, output_dir / "summary.md")
        print(json.dumps({"output_dir": str(output_dir), **verdict}, ensure_ascii=False))
        return 2 if args.strict and not verdict["credible"] else 0
    finally:
        if sink is not None:
            sink.close()
        knowledge.close()


if __name__ == "__main__":
    raise SystemExit(main())
