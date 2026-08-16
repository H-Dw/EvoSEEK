import json

from fitness_agents.contracts.schemas import FitnessObservation, Prediction
from fitness_agents.data import load_dataset_bundle
from fitness_agents.knowledge import KnowledgeEngine


def test_knowledge_channels_are_independently_switchable(experiment_config, tmp_path):
    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path, experiment_config.task.oracle_data_path
    )
    engine = KnowledgeEngine(
        experiment_config.knowledge, graph_path=tmp_path / "kg.sqlite", assay_id="test"
    )
    engine.update(bundle.initial_variants, bundle.initial_observations)
    evidence = engine.evidence_for(bundle.oracle_pool[:2], round_id=1)
    assert {item.channel for item in evidence[bundle.oracle_pool[0].variant_id]} == {
        "physchem", "conservation", "structure", "kg"
    }
    assert engine.evidence_for(bundle.oracle_pool[:2], round_id=1, delete_evidence=True) == {}
    assert all(edge["predicate"] == "OBSERVED_IN_CONTEXT" for edge in engine.graph.export_edges())
    engine.close()


def test_agent_graph_tool_fuses_typed_evidence_and_enforces_round_visibility(
    experiment_config, tmp_path
):
    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path, experiment_config.task.oracle_data_path
    )
    engine = KnowledgeEngine(
        experiment_config.knowledge, graph_path=tmp_path / "agent-kg.sqlite", assay_id="test"
    )
    engine.update(bundle.initial_variants, bundle.initial_observations)
    candidates = bundle.oracle_pool[:3]
    predictions = [
        Prediction(
            variant_id=variant.variant_id,
            fitness_mean=0.8 - index * 0.1,
            fitness_std=0.05,
            interval_90=(0.6, 1.0),
            ood_score=0.1,
            component_scores={"sequence": 0.7, "structure": 0.4},
            model_version="test-model:v1",
        )
        for index, variant in enumerate(candidates)
    ]
    evidence = engine.evidence_for(candidates, round_id=1)
    engine.record_inference_context(candidates, predictions, evidence, round_id=1)

    hidden_current_round = FitnessObservation(
        variant_id=candidates[0].variant_id,
        fitness=98765.4321,
        split_role="oracle_pool",
        round_revealed=1,
    )
    engine.update([candidates[0]], [hidden_current_round])
    tool = engine.agent_tool(max_rows=2)
    context = tool.hypothesis_context(round_id=1)

    assert context["tool"] == "knowledge_graph"
    assert len(context["current_candidate_predictions"]) == 2
    assert {item["source_type"] for item in context["current_candidate_predictions"]} == {
        "model_prediction"
    }
    assert context["top_knowledge_evidence"]
    assert {item["source_type"] for item in context["top_knowledge_evidence"]} == {
        "computed_evidence"
    }
    assert "98765.4321" not in json.dumps(context)
    assert engine.graph.export_agent_queries()[0]["query_id"] == context["query_id"]
    predicates = {edge["predicate"] for edge in engine.graph.export_edges()}
    assert {"OBSERVED_IN_CONTEXT", "PREDICTED_AS", "SUPPORTED_BY_EVIDENCE"} <= predicates
    engine.close()


def test_hypothesis_context_includes_persisted_evidence_before_predictions(
    experiment_config, tmp_path
):
    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path, experiment_config.task.oracle_data_path
    )
    engine = KnowledgeEngine(
        experiment_config.knowledge,
        graph_path=tmp_path / "pre-hypothesis-kg.sqlite",
        assay_id="test",
    )
    engine.update(bundle.initial_variants, bundle.initial_observations)
    observed = bundle.initial_variants[:3]
    evidence = engine.evidence_for(observed, round_id=1)
    engine.graph.add_variants(observed)
    engine.graph.add_evidence([item for items in evidence.values() for item in items])
    context = engine.agent_tool(max_rows=4).hypothesis_context(round_id=1)
    assert context["current_candidate_predictions"] == []
    assert context["top_knowledge_evidence"]
    assert all(
        item["source_type"] == "computed_evidence" for item in context["top_knowledge_evidence"]
    )
    engine.close()
