from __future__ import annotations

import json

from common import (
    ensure,
    load_config,
    make_observations,
    make_predictions,
    parse_args,
    resolve_output,
    variant_grid,
    write_result,
)

from fitness_agents.config import KnowledgeConfig
from fitness_agents.contracts.schemas import Evidence, FitnessObservation
from fitness_agents.knowledge import KnowledgeEngine


class CustomEvidenceProvider:
    channel = "custom_assay_prior"

    def evaluate(self, variant, *, round_id):
        return Evidence(
            evidence_id=f"ev:custom:{round_id}:{variant.variant_id[-8:]}",
            variant_id=variant.variant_id,
            channel=self.channel,
            statement="Custom provider contract exercised.",
            score=0.25,
            source_id="module-test:custom-provider",
            confidence=0.5,
            round_id=round_id,
        )


def main() -> None:
    args = parse_args("configs/module_tests/knowledge_runtime.yaml")
    config = load_config(args.config)
    output = resolve_output(config, args.output_dir)
    knowledge_raw = dict(config["knowledge"])
    knowledge_raw["site_profiles"] = {
        int(position): profile
        for position, profile in knowledge_raw.get("site_profiles", {}).items()
    }
    engine = KnowledgeEngine(
        KnowledgeConfig(**knowledge_raw),
        graph_path=output / "knowledge.sqlite",
        assay_id=str(config["assay_id"]),
    )
    engine.register_provider(CustomEvidenceProvider())

    variants = variant_grid()
    observed = variants[:16]
    candidates = variants[16:22]
    observations = make_observations(observed, round_revealed=0)
    engine.update(observed, observations)
    round_id = int(config["query_round"])
    evidence = engine.evidence_for(candidates, round_id=round_id)
    expected_channels = {"physchem", "conservation", "structure", "kg", "custom_assay_prior"}
    ensure(
        {item.channel for item in evidence[candidates[0].variant_id]} == expected_channels,
        "Configured evidence channels were not all exercised",
    )
    scores = engine.scores(evidence)
    ensure(set(scores) == {item.variant_id for item in candidates}, "Evidence scores are incomplete")

    predictions = make_predictions(candidates, model_version="knowledge-module:v1")
    engine.record_inference_context(
        candidates,
        predictions,
        evidence,
        round_id=round_id,
        intervention_tags=("module_test",),
    )
    all_evidence_ids = [item.evidence_id for bundle in evidence.values() for item in bundle]
    engine.graph.add_hypothesis(
        "hyp:prior-round",
        0,
        "Prior hypothesis retained for history query coverage.",
        all_evidence_ids[:2],
        status="active",
    )

    hidden_value = 98765.4321
    engine.update(
        [candidates[0]],
        [
            FitnessObservation(
                candidates[0].variant_id,
                hidden_value,
                "oracle_pool",
                round_id,
                "hidden_current_round",
            )
        ],
    )
    tool = engine.agent_tool(max_rows=int(config["max_query_rows"]))
    context = tool.hypothesis_context(round_id=round_id)
    explanation = tool.explain_variant(candidates[0].variant_id, round_id=round_id)
    ensure(context["tool"] == "knowledge_graph", "Safe KG tool identity is missing")
    ensure(
        str(hidden_value) not in json.dumps(context, sort_keys=True),
        "Current-round hidden measurement leaked into hypothesis context",
    )
    ensure(
        len(context["current_candidate_predictions"]) <= int(config["max_query_rows"]),
        "KG query row bound was ignored",
    )
    ensure(explanation["found"], "Variant explanation did not find a recorded candidate")
    ensure(engine.evidence_for(candidates, round_id=round_id, delete_evidence=True) == {}, "Ablation failed")

    edges = engine.graph.export_edges()
    queries = engine.graph.export_agent_queries()
    current_query_ids = {context["query_id"], explanation["query_id"]}
    audited_query_ids = {item["query_id"] for item in queries}
    predicates = {str(item["predicate"]) for item in edges}
    ensure(
        {"OBSERVED_IN_CONTEXT", "PREDICTED_AS", "SUPPORTED_BY_EVIDENCE"}.issubset(predicates),
        "Typed observation/prediction/evidence edges are incomplete",
    )
    ensure(
        current_query_ids.issubset(audited_query_ids),
        "The queries issued by this run are missing from the audit log",
    )
    engine.close()

    write_result(
        output,
        "knowledge_runtime",
        {
            "config": config["_config_path"],
            "channels": sorted(expected_channels),
            "candidate_scores": scores,
            "query_ids": sorted(current_query_ids),
            "audit_log_size": len(queries),
            "edge_count": len(edges),
            "predicates": sorted(predicates),
            "round_visibility_guard": True,
        },
    )


if __name__ == "__main__":
    main()
