import json

import pytest

from fitness_agents.contracts.schemas import (
    CriterionResult,
    CriterionSignal,
    FitnessObservation,
    HypothesisAssessment,
    HypothesisReflection,
    HypothesisStatus,
    Prediction,
    ValidationRecord,
)
from fitness_agents.data import load_dataset_bundle
from fitness_agents.knowledge import KnowledgeEngine
from fitness_agents.knowledge.graph import ObservationKnowledgeGraph


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


def test_validation_priors_append_and_apply_fidelity_and_recency_weights(
    experiment_config, tmp_path
):
    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path, experiment_config.task.oracle_data_path
    )
    variant = bundle.oracle_pool[0]
    engine = KnowledgeEngine(
        experiment_config.knowledge,
        graph_path=tmp_path / "validation-prior-kg.sqlite",
        assay_id="test",
    )
    engine.graph.add_variants([variant])
    records = (
        ValidationRecord(
            "wet:1", variant.variant_id, 1, "wet", variant.mutation_notation,
            1.0, 0.0, "assay", None, 1.0, 1.0, "reason", "h1", "ha1", (),
        ),
        ValidationRecord(
            "dry:1", variant.variant_id, 1, "dry", variant.mutation_notation,
            0.8, 0.2, "predictor", "model:v1", 0.2, 0.5, "reason", "h1", "ha1", (),
        ),
    )
    engine.record_validation(records)
    engine.record_validation(records)

    next_round = engine.graph.validation_prior_context(round_id=2, limit=10)
    later_round = engine.graph.validation_prior_context(round_id=3, limit=10)

    assert len(next_round) == 2
    weights = {item["validation_type"]: item["effective_weight"] for item in next_round}
    later_weights = {
        item["validation_type"]: item["effective_weight"] for item in later_round
    }
    assert weights == {"dry": 0.1, "wet": 1.0}
    assert later_weights["wet"] == pytest.approx(0.85)
    assert later_weights["dry"] == pytest.approx(0.085)
    engine.close()


def test_hypothesis_memory_is_separate_from_sample_validation_rows(tmp_path) -> None:
    graph = ObservationKnowledgeGraph(
        tmp_path / "hypothesis-memory.sqlite",
        assay_id="test",
        recency_decay=0.85,
        wild_type_code="A",
        mutable_positions=(1,),
    )
    graph.add_hypothesis("h1", 1, "A bounded testable claim.", ("e1",))
    assessment = HypothesisAssessment(
        "ha1",
        "h1",
        "fs1",
        1,
        HypothesisStatus.CONTRADICTED,
        (
            CriterionResult(
                "c1",
                CriterionSignal.CONTRADICT,
                -0.2,
                0.0,
                -0.2,
                ("v1",),
                "ok",
                "mean_delta",
                "v1",
                "target_below_control",
            ),
        ),
        ("v1",),
        ("c1",),
        (),
        "v1",
    )
    reflection = HypothesisReflection(
        "hr1",
        "h1",
        "ha1",
        1,
        "CONTRADICTED",
        "The tested assumption was contradicted.",
        (),
        ("The target edit direction is not reusable.",),
        ("The matched alternative remains untested.",),
        ("test_matched_alternative",),
        ("v1",),
        ("e1",),
        "mock_rethink",
    )
    graph.add_hypothesis_assessment(assessment)
    graph.add_hypothesis_reflection(reflection)

    context = graph.agent_hypothesis_context(round_id=2, limit=5)

    assert context["validation_prior"] == []
    assert len(context["prior_hypothesis_memory"]) == 1
    memory = context["prior_hypothesis_memory"][0]
    assert memory["assessment_status"] == "CONTRADICTED"
    assert memory["invalidated_assumptions"] == [
        "The target edit direction is not reusable."
    ]
    assert memory["selection_eligible"] is False
    status = graph.connection.execute(
        "SELECT status FROM hypotheses WHERE hypothesis_id = 'h1'"
    ).fetchone()[0]
    assert status == "CONTRADICTED"
    graph.close()


def test_evidence_for_can_score_kg_without_static_channels(experiment_config, tmp_path):
    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path, experiment_config.task.oracle_data_path
    )
    engine = KnowledgeEngine(
        experiment_config.knowledge, graph_path=tmp_path / "kg-only.sqlite", assay_id="test"
    )
    engine.update(bundle.initial_variants, bundle.initial_observations)
    remaining = bundle.oracle_pool[:5]
    kg_only = engine.evidence_for(remaining, round_id=1, channels=("kg",))
    assert all(
        {item.channel for item in bundle_items} == {"kg"}
        for bundle_items in kg_only.values()
    )
    assert engine.site_feature_tables() == {}
    engine.close()
