from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from fitness_agents.agents.llm import (
    HYPOTHESIS_SCHEMA,
    build_scientist_hypothesis_messages,
    create_llm_client,
    load_scientist_profile,
)
from fitness_agents.agents.scientist import ScientistAgent
from fitness_agents.config import load_experiment_config
from fitness_agents.contracts.agent_io import ScientistContextInput
from fitness_agents.contracts.schemas import CampaignState
from fitness_agents.data import load_dataset_bundle
from fitness_agents.kg_interaction.contracts import (
    InteractionResult,
    KGQueryContext,
    KGQueryStep,
    QueryIntent,
)
from fitness_agents.kg_interaction.operators import (
    HypothesisContextOperator,
    LocalKnowledgeQueryOperator,
)
from fitness_agents.knowledge.engine import KnowledgeEngine
from fitness_agents.protein_features import ProteinTaskContext

SYSTEM_LOCAL_QUERY = (
    "general protein structure stability binding mutation physicochemical epistasis knowledge"
)
SYSTEM_LOCAL_ANCHORS = (
    "protein structure and stability",
    "binding interface mutation effects",
    "physicochemical substitution mechanisms",
    "epistasis and residue interactions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real Qwen RAG-to-KG path and capture the Scientist prompt."
    )
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/knowledge_agent_qwen_unified_api.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/rag_api_hypothesis_simulation",
    )
    parser.add_argument("--call-llm", action="store_true")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def local_kg_records(snapshot: Any) -> dict[str, Any]:
    local_entities = [
        item
        for item in snapshot.entities
        if item.entity_type
        in {
            "Document",
            "DocumentChunk",
            "Claim",
            "Publication",
            "CitationSupport",
            "Evidence",
        }
        and (
            item.source_group == "directed_evolution_library"
            or item.entity_type in {"Publication", "CitationSupport"}
            or item.entity_id.startswith(
                ("doc:", "chunk:", "claim:", "evidence:ev:local_rag:")
            )
        )
    ]
    entity_ids = {item.entity_id for item in local_entities}
    local_relations = [
        item
        for item in snapshot.relations
        if item.subject_id in entity_ids or item.object_id in entity_ids
    ]
    return {"entities": local_entities, "relations": local_relations}


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config(args.experiment_config)
    if config.task.public_data_path is None or config.task.oracle_data_path is None:
        raise RuntimeError("This simulation requires the configured legacy demo dataset bundle")
    dataset = load_dataset_bundle(config.task.public_data_path, config.task.oracle_data_path)
    task_context = ProteinTaskContext.from_task(config.task)
    state = CampaignState(
        run_id="qwen-rag-hypothesis-simulation",
        mode="knowledge_agent",
        seed=config.seed,
        round_id=1,
        observed=list(dataset.initial_observations),
    )
    engine = KnowledgeEngine(
        config.knowledge,
        graph_path=output_dir / "knowledge_graph.sqlite",
        structured_graph_path=output_dir / "structured_kg.sqlite",
        assay_id=config.task.assay_id,
        protein_id=config.task.protein_id,
        task_context=task_context,
        protein_name=config.task.protein_name,
        protein_aliases=config.task.protein_aliases,
        protein_accessions=config.task.protein_accessions,
    )
    try:
        engine.update(dataset.initial_variants, dataset.initial_observations)
        prefetch_result, prefetch_evidence = engine.prefetch_local_knowledge(
            round_id=1,
            objective=config.task.objective,
            assay_conditions=config.task.assay_conditions,
            anchors=tuple(sorted(engine.providers)),
        )
        if prefetch_result is None:
            raise RuntimeError("Local knowledge prefetch did not return a result")
        engine.sync_structured_kg(
            run_id=state.run_id,
            round_id=1,
            variants=dataset.initial_variants,
            observations=dataset.initial_observations,
            evidence=prefetch_evidence,
        )

        query_context = KGQueryContext(
            run_id=state.run_id,
            round_id=1,
            allowed_variant_ids=frozenset(item.variant_id for item in dataset.initial_variants),
            max_rows=config.kg_interaction.max_rows,
        )
        context_pack = HypothesisContextOperator(engine.agent_tool()).execute(
            KGQueryStep(
                "context",
                "hypothesis_context",
                QueryIntent.CONTEXT,
                {"limit": config.kg_interaction.max_rows},
            ),
            query_context,
        )
        local_pack = LocalKnowledgeQueryOperator(engine).execute(
            KGQueryStep(
                "local_knowledge",
                "query_local_knowledge",
                QueryIntent.SUPPORT,
                {
                    "query": SYSTEM_LOCAL_QUERY,
                    "anchors": list(SYSTEM_LOCAL_ANCHORS),
                    "limit": config.kg_interaction.max_rows,
                },
                ("context",),
            ),
            query_context,
        )
        interaction = InteractionResult(
            plan_id="hypothesis-evidence:r1",
            packs=(context_pack, local_pack),
            executed_steps=("context", "local_knowledge"),
            skipped_steps=(),
            stop_reason="simulation_plan_complete",
        )

        final_structured = engine.sync_structured_kg(
            run_id=state.run_id,
            round_id=1,
            variants=dataset.initial_variants,
            observations=dataset.initial_observations,
            evidence=engine.local_evidence(round_id=1),
        )
        llm = create_llm_client(
            config.llm.provider,
            profile=config.llm.profile,
            model=config.llm.model,
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            reasoning_effort=config.llm.reasoning_effort,
            thinking=config.llm.thinking,
        ) if args.call_llm else None
        agent = ScientistAgent(
            llm if llm is not None else create_llm_client("mock"),
            task_context=task_context,
            objective=config.task.objective,
            knowledge_graph=engine.agent_tool(),
        )
        context = agent.sanitized_context(
            state, dataset.initial_variants, dataset.initial_observations
        )
        context["kg_interaction"] = asdict(interaction)
        validated_context = ScientistContextInput.model_validate(context)
        prompt_messages = build_scientist_hypothesis_messages(
            profile=load_scientist_profile(config.llm.profile),
            sanitized_context=validated_context,
            evidence=engine.local_evidence(round_id=1),
            output_schema=HYPOTHESIS_SCHEMA,
        )
        hypothesis = None
        if args.call_llm:
            hypothesis = agent.propose_hypothesis(
                state,
                dataset.initial_variants,
                dataset.initial_observations,
                engine.local_evidence(round_id=1),
                kg_interaction=interaction,
            )

        write_json(output_dir / "rag_prefetch_retrieval.json", prefetch_result)
        write_json(output_dir / "rag_prefetch_evidence.json", prefetch_evidence)
        write_json(output_dir / "kg_interaction.json", interaction)
        write_json(output_dir / "kg_materialization.json", local_kg_records(final_structured.snapshot))
        write_json(output_dir / "scientist_prompt.json", {"messages": prompt_messages})
        write_json(output_dir / "hypothesis.json", hypothesis)
        write_json(
            output_dir / "run_manifest.json",
            {
                "experiment_config": str(Path(args.experiment_config).resolve()),
                "local_corpus_root": [str(item.path) for item in config.knowledge.local_knowledge.roots],
                "corpus_index_path": config.knowledge.local_knowledge.corpus_index_path,
                "retrieval_overlay_path": config.knowledge.local_knowledge.retrieval_overlay_path,
                "embedding_backend": engine.local_knowledge.embedding_backend.fingerprint,
                "reranker_backend": engine.local_knowledge.reranker_backend.fingerprint,
                "build_report": engine.local_knowledge_build_report,
                "embedding_row_count": int(
                    engine.local_knowledge.index.connection.execute(
                        "SELECT COUNT(*) FROM embeddings"
                    ).fetchone()[0]
                ),
                "query": SYSTEM_LOCAL_QUERY,
                "prefetch_chunk_count": len(prefetch_result.chunks),
                "interaction_fact_count": local_pack.fact_count,
                "structured_local_entity_count": len(local_kg_records(final_structured.snapshot)["entities"]),
                "structured_local_relation_count": len(local_kg_records(final_structured.snapshot)["relations"]),
                "llm_called": bool(args.call_llm),
            },
        )
    finally:
        engine.close()
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
