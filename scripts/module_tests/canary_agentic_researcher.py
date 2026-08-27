#!/usr/bin/env python3
"""Run one real, evidence-only Researcher + Qwen canary without a fitness benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fitness_agents.agents.researcher import NativeResearcherClient
from fitness_agents.config import load_experiment_config
from fitness_agents.contracts.researcher import (
    ExternalRetrievalPlan,
    FeatureEvidencePlan,
    ResearcherAssayContext,
    ResearcherContextInput,
    ResearcherFacetCatalog,
    ResearcherKnowledgeRecordCard,
    ResearcherRoundReceipt,
    ResearcherSampleCard,
    ResearcherToolCard,
)
from fitness_agents.contracts.schemas import FitnessObservation, Variant
from fitness_agents.kg_interaction.researcher import (
    ResearcherPlanningController,
    stable_payload_hash,
)
from fitness_agents.knowledge.engine import KnowledgeEngine
from fitness_agents.protein_features import ProteinTaskContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/gb1_3features_agentic_researcher_deepseek_v4_pro.yaml"
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


def _record_card(record: Any) -> ResearcherKnowledgeRecordCard:
    return ResearcherKnowledgeRecordCard(
        record_id=record.record_id,
        record_type=record.record_type,
        retrieval_text=record.retrieval_text,
        knowledge_type=record.knowledge_type,
        permission=record.permission,
        scientific_quality=record.scientific_quality,
        task_applicability=record.task_applicability,
        boundary_conditions=record.boundary_conditions,
        counterclaims=record.counterclaims,
        abstain_if=record.abstain_if,
        facets=record.facets,
    )


def _tools(allowed_positions: tuple[int, ...]) -> tuple[ResearcherToolCard, ...]:
    return (
        ResearcherToolCard(
            tool_id="query_physchem_delta",
            channel="physchem",
            allowed_positions=allowed_positions,
            allowed_focus=("site_deltas", "global_sequence_deltas", "special_flags"),
        ),
        ResearcherToolCard(
            tool_id="query_evolutionary_profile",
            channel="conservation",
            allowed_positions=allowed_positions,
            allowed_focus=("site_log_odds", "pairwise_signal", "profile_quality"),
        ),
        ResearcherToolCard(
            tool_id="query_structure_environment",
            channel="structure",
            allowed_positions=allowed_positions,
            allowed_focus=(
                "solvent_exposure",
                "contact_geometry",
                "interface_contacts",
                "backbone_geometry",
                "interaction_flags",
            ),
        ),
    )


def run_canary(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite canary output: {output_root}")
    output_root.mkdir(parents=True)
    task_context = ProteinTaskContext.from_task(config.task)
    engine = KnowledgeEngine(
        config.knowledge,
        graph_path=output_root / "measurement_kg.sqlite",
        structured_graph_path=output_root / "structured_kg.sqlite",
        assay_id="generic-binding-canary",
        protein_id=config.task.protein_id,
        validation_config=config.validation,
        task_context=task_context,
        protein_name=config.task.protein_name,
        protein_aliases=config.task.protein_aliases,
        protein_accessions=config.task.protein_accessions,
        local_knowledge_enabled=True,
    )
    try:
        wt = Variant(
            variant_id="VISIBLE-REFERENCE",
            variant=config.task.wild_type_sites,
            sequence=task_context.full_sequence,
            mutation_notation="WT",
            mutation_count=0,
            split_role="canary_reference",
        )
        observation = FitnessObservation(
            variant_id=wt.variant_id,
            fitness=0.0,
            split_role="canary_reference",
            round_revealed=0,
            source="synthetic_canary_measurement",
        )
        engine.update((wt,), (observation,))
        feature_evidence = engine.evidence_for((wt,), round_id=1)[wt.variant_id]
        engine.graph.add_evidence(feature_evidence)

        researcher = NativeResearcherClient(
            provider=config.researcher.provider or config.llm.provider,
            model=config.researcher.model or config.llm.model,
            base_url=config.researcher.base_url or config.llm.base_url,
            api_key=config.researcher.api_key or config.llm.api_key,
            profile=config.researcher.profile,
            temperature=config.researcher.temperature,
            max_tokens=config.researcher.max_tokens,
            reasoning_effort=config.researcher.reasoning_effort,
            thinking=config.researcher.thinking,
            max_input_chars=config.researcher.max_input_chars,
            request_timeout_seconds=(
                config.researcher.request_timeout_seconds
                or config.llm.request_timeout_seconds
            ),
            max_transport_retries=config.llm.max_transport_retries,
            max_truncation_retries=config.llm.max_truncation_retries,
            max_syntax_retries=config.llm.max_syntax_retries,
            max_schema_retries=config.llm.max_schema_retries,
            max_semantic_retries=config.llm.max_semantic_retries,
            retry_backoff_seconds=config.llm.retry_backoff_seconds,
        )
        facet_catalog = engine.local_knowledge.index.facet_catalog()  # type: ignore[union-attr]
        forbidden_terms = tuple(
            item
            for item in (
                config.task.protein_id,
                config.task.protein_name or "",
                *config.task.protein_aliases,
                *config.task.protein_accessions,
            )
            if item
        )
        controller = ResearcherPlanningController(
            config.researcher,
            mutable_positions=config.task.mutable_positions,
            facet_catalog=facet_catalog,
            enabled_feature_channels=config.kg_interaction.feature_channels,
            forbidden_query_terms=forbidden_terms,
        )
        researcher.bind_external_plan_validator(controller.validate_external_plan)
        assay = ResearcherAssayContext(
            assay_id="A1",
            objective=(
                "Audit whether one visible reference measurement can support a mechanism "
                "transfer, identify the external evidence boundary, and request only the "
                "quality or limitation projections needed to assess applicability."
            ),
            fitness_scale="synthetic reference-only value with no benchmark interpretation",
            optimization_direction="higher_is_better",
            conditions={"scope": "non-pathogen protein engineering evidence audit"},
        )
        sample = ResearcherSampleCard(
            sample_id="S1",
            observation_id="O1",
            measured_fitness=0.0,
            round_revealed=0,
            source="synthetic_canary_measurement",
            mutated_positions=(),
        )
        phase_a_context = ResearcherContextInput(
            phase="external_retrieval",
            run_id="agentic-researcher-live-canary",
            round_id=1,
            task=assay.objective,
            assay=assay,
            measurement_kg=(sample,),
            facet_catalog=ResearcherFacetCatalog(allowed_values=facet_catalog),
        )
        external_plan = researcher.plan_external(phase_a_context)
        needs = controller.validate_external_plan(external_plan)
        (output_root / "phase-a-plan.json").write_text(
            json.dumps(
                {
                    "context_hash": stable_payload_hash(phase_a_context),
                    "plan": _jsonable(external_plan),
                    "validated_needs": [_jsonable(item) for item in needs],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        if not needs:
            raise RuntimeError("Live canary requires a non-empty compliant Phase A plan")

        query_results = []
        accepted_record_ids: set[str] = set()
        staged_results = []
        staged_evidence = []
        record_cards: list[ResearcherKnowledgeRecordCard] = []
        for need in needs:
            result, _ = engine.retrieve_local_knowledge(
                query=need.query,
                intent=need.intent,
                round_id=1,
                anchors=(),
                top_k=need.top_k,
                facets=need.facets,
                stage=False,
            )
            query_results.append(result)
            accepted_records = []
            accepted_chunk_ids: set[str] = set()
            for record in result.records:
                if record.record_id in accepted_record_ids:
                    continue
                if len(accepted_record_ids) >= config.researcher.max_retrieved_records:
                    break
                accepted_record_ids.add(record.record_id)
                accepted_records.append(record)
                accepted_chunk_ids.update(record.evidence_chunk_ids)
                record_cards.append(_record_card(record))
            accepted_chunks = tuple(
                item for item in result.chunks if item.chunk_id in accepted_chunk_ids
            )
            accepted_claims = tuple(
                item
                for item in result.claims
                if set(item.evidence_chunk_ids).intersection(accepted_chunk_ids)
            )
            staged = replace(
                result,
                chunks=accepted_chunks,
                claims=accepted_claims,
                records=tuple(accepted_records),
            )
            staged_results.append(staged)
            staged_evidence.extend(
                engine.local_knowledge.evidence_from_result(staged)  # type: ignore[union-attr]
            )
            (output_root / "phase-a-execution.json").write_text(
                json.dumps(
                    [
                        {
                            "query_id": item.query_id,
                            "sanitized_query": item.sanitized_query,
                            "policy_decision": item.policy_decision,
                            "record_ids": [record.record_id for record in item.records],
                            "chunk_ids": [chunk.chunk_id for chunk in item.chunks],
                            "warnings": list(item.warnings),
                        }
                        for item in staged_results
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if not record_cards:
            raise RuntimeError("Qwen retrieval returned no native evidence record")
        engine.stage_local_knowledge(
            round_id=1,
            results=staged_results,
            evidence=staged_evidence,
        )
        engine.sync_structured_kg(
            run_id="agentic-researcher-live-canary",
            round_id=1,
            variants=(wt,),
            observations=(observation,),
            evidence=(*feature_evidence, *staged_evidence),
        )

        phase_b_context = ResearcherContextInput(
            phase="feature_evidence",
            run_id="agentic-researcher-live-canary",
            round_id=1,
            task=assay.objective,
            assay=assay,
            measurement_kg=(sample,),
            sample_map=(sample,),
            rag_records=tuple(record_cards),
            facet_catalog=ResearcherFacetCatalog(allowed_values=facet_catalog),
            tool_catalog=_tools(tuple(config.task.mutable_positions)),
        )
        feature_plan = researcher.plan_features(phase_b_context)
        (output_root / "phase-b-plan.json").write_text(
            json.dumps(
                {
                    "context_hash": stable_payload_hash(phase_b_context),
                    "visible_record_ids": [item.record_id for item in record_cards],
                    "plan": _jsonable(feature_plan),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        steps = controller.validate_feature_plan(
            feature_plan,
            sample_id_to_variant_id={"S1": wt.variant_id},
        )
        (output_root / "phase-b-execution-plan.json").write_text(
            json.dumps(
                {
                    "context_hash": stable_payload_hash(phase_b_context),
                    "visible_record_ids": [item.record_id for item in record_cards],
                    "plan": _jsonable(feature_plan),
                    "validated_steps": [_jsonable(item) for item in steps],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        if not steps:
            raise RuntimeError("Live canary requires a non-empty compliant Phase B plan")

        tool = engine.agent_tool(max_rows=config.kg_interaction.max_rows)
        channel_by_operator = {
            "query_physchem_delta": "physchem",
            "query_evolutionary_profile": "conservation",
            "query_structure_environment": "structure",
        }
        tool_results = []
        for step in steps:
            tool_results.append(
                tool.feature_evidence(
                    step.arguments["variant_id"],
                    channel=channel_by_operator[step.operator],
                    round_id=1,
                    projection=step.arguments["projection"],
                    positions=step.arguments["positions"],
                )
            )

        receipt = ResearcherRoundReceipt(
            run_id="agentic-researcher-live-canary",
            round_id=1,
            profile=config.researcher.profile,
            profile_hash=researcher.profile_hash,
            external_schema_hash=researcher.schema_hash(ExternalRetrievalPlan),
            feature_schema_hash=researcher.schema_hash(FeatureEvidencePlan),
            kg_snapshot_hash=stable_payload_hash(
                {
                    "phase_a": stable_payload_hash(phase_a_context),
                    "phase_b": stable_payload_hash(phase_b_context),
                }
            ),
            external_plan=external_plan,
            feature_plan=feature_plan,
            query_ids=tuple(item.query_id for item in query_results),
            record_ids=tuple(item.record_id for item in record_cards),
            tool_query_ids=tuple(item["query_id"] for item in tool_results),
            budget_used={
                "rag_queries": len(query_results),
                "retrieved_records": len(record_cards),
                "feature_requests": len(tool_results),
            },
        )
        report = {
            "schema_version": "agentic-researcher-live-canary:v1",
            "status": "passed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "fixture": {
                "measurement_count": 1,
                "reference_only": True,
                "mutation_identity_exposed": False,
                "benchmark_truth_used": False,
                "fitness_benchmark_run": False,
                "validation_selection_run": False,
                "pathogen_material_used": False,
            },
            "providers": {
                "researcher": {
                    "provider": config.researcher.provider or config.llm.provider,
                    "model": researcher.model,
                    "profile": config.researcher.profile,
                    "profile_version": researcher.profile_version,
                },
                "retrieval": {
                    "embedding": engine.local_knowledge.embedding_backend.name,  # type: ignore[union-attr]
                    "reranker": engine.local_knowledge.reranker_backend.name,  # type: ignore[union-attr]
                },
            },
            "index": engine.local_knowledge.index.stats(),  # type: ignore[union-attr]
            "phase_a": {
                "context_hash": stable_payload_hash(phase_a_context),
                "plan": _jsonable(external_plan),
                "executed": [
                    {
                        "query_id": item.query_id,
                        "sanitized_query": item.sanitized_query,
                        "record_ids": [record.record_id for record in item.records],
                        "warnings": list(item.warnings),
                    }
                    for item in staged_results
                ],
            },
            "phase_b": {
                "context_hash": stable_payload_hash(phase_b_context),
                "visible_record_ids": [item.record_id for item in record_cards],
                "plan": _jsonable(feature_plan),
                "executed_tools": [
                    {
                        "query_id": item["query_id"],
                        "channel": item["channel"],
                        "projection": item["projection"],
                        "positions": item["positions"],
                        "evidence_count": len(item["evidence"]),
                    }
                    for item in tool_results
                ],
            },
            "budgets": {
                "max_input_chars_per_phase": config.researcher.max_input_chars,
                "max_output_tokens_per_phase": config.researcher.max_tokens,
                "max_rag_queries": config.researcher.max_rag_queries,
                "top_k_per_query": config.researcher.rag_top_k_per_query,
                "max_records": config.researcher.max_retrieved_records,
                "max_visible_samples": config.researcher.max_feature_variants,
                "max_feature_requests": config.researcher.max_feature_requests,
            },
            "receipt": _jsonable(receipt),
            "integrity": {
                "two_distinct_researcher_phases": True,
                "opaque_assay_label": phase_a_context.assay.assay_id == "A1",
                "identity_neutral_queries": all(
                    not any(
                        term.casefold() in need.scientific_question.casefold()
                        for term in forbidden_terms
                    )
                    for need in external_plan.needs
                ),
                "query_execution_count_matches_plan": len(query_results) == len(needs),
                "native_record_types_retained": sorted(
                    {item.record_type for item in record_cards}
                ),
                "permission_values": sorted({item.permission for item in record_cards}),
                "feature_execution_allowlisted": all(
                    step.operator in channel_by_operator for step in steps
                ),
                "hidden_reasoning_recorded": False,
            },
        }
        (output_root / "researcher-round-receipt.json").write_text(
            json.dumps(_jsonable(receipt), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_root / "canary-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        engine.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        PROJECT_ROOT / "artifacts/canary" / f"agentic-researcher-{timestamp}"
    )
    try:
        report = run_canary(args.config.resolve(), output.resolve())
    except Exception as error:  # noqa: BLE001 - live provider boundary is audited
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": "agentic-researcher-live-canary:v1",
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error)[:800],
            "failure_policy": "abort_round",
            "fixed_query_fallback_used": False,
        }
        (output / "canary-failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({**failure, "output": str(output.resolve())}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output.resolve()),
                "rag_queries": report["receipt"]["budget_used"]["rag_queries"],
                "records": report["receipt"]["budget_used"]["retrieved_records"],
                "feature_requests": report["receipt"]["budget_used"]["feature_requests"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
