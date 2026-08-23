from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fitness_agents.contracts.schemas import (
    CriterionResult,
    CriterionSignal,
    HypothesisAssessment,
    HypothesisReflection,
    HypothesisStatus,
    ReThinkReflection,
    ValidationRecord,
    Variant,
)
from fitness_agents.knowledge import KnowledgeEngine
from fitness_agents.knowledge.graph import ObservationKnowledgeGraph

LEGACY_VALIDATION_COLUMNS = {
    "reflection_id",
    "reflection_verdict",
    "reflection_summary",
    "reflection_quality_status",
    "reflection_advisory_only",
}


def _create_legacy_main_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE variants (
                variant_id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                sequence TEXT NOT NULL DEFAULT '',
                mutation_notation TEXT NOT NULL,
                mutation_count INTEGER NOT NULL DEFAULT 0,
                split_role TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE validation_records (
                record_id TEXT PRIMARY KEY,
                variant_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                validation_type TEXT NOT NULL CHECK(validation_type IN ('wet', 'dry')),
                mutation_notation TEXT NOT NULL,
                value REAL NOT NULL,
                uncertainty REAL NOT NULL,
                source_id TEXT NOT NULL,
                model_version TEXT,
                base_weight REAL NOT NULL,
                reliability REAL NOT NULL,
                agent_reason TEXT NOT NULL,
                hypothesis_id TEXT,
                evidence_ids_json TEXT NOT NULL,
                reflection_id TEXT,
                reflection_verdict TEXT,
                reflection_summary TEXT NOT NULL,
                reflection_quality_status TEXT,
                reflection_advisory_only INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (variant_id) REFERENCES variants(variant_id)
            );
            """
        )
        connection.commit()
    finally:
        connection.close()


def _graph(path: Path) -> ObservationKnowledgeGraph:
    return ObservationKnowledgeGraph(
        path,
        assay_id="compatibility-assay",
        recency_decay=0.85,
        wild_type_code="A",
        mutable_positions=(1,),
    )


def _variant() -> Variant:
    return Variant("v1", "W", "W", "A1W", 1, "oracle_pool")


def _validation_record() -> ValidationRecord:
    return ValidationRecord(
        record_id="VR01-001",
        variant_id="v1",
        round_id=1,
        validation_type="wet",
        mutation_notation="A1W",
        value=0.9,
        uncertainty=0.0,
        source_id="wet:benchmark",
        model_version=None,
        base_weight=1.0,
        reliability=1.0,
        agent_reason="legacy sample rationale",
        hypothesis_id="h1",
        evidence_ids=("e1",),
        reflection_id="r1",
        reflection_verdict="support",
        reflection_summary="The measured value supports the sample rationale.",
        reflection_quality_status="model",
        reflection_advisory_only=True,
        assessment_id="ha1",
    )


@pytest.mark.parametrize("legacy_main_schema", [False, True])
def test_validation_schema_is_additive_for_fresh_and_legacy_main_databases(
    tmp_path: Path,
    legacy_main_schema: bool,
) -> None:
    path = tmp_path / "observation-kg.sqlite"
    if legacy_main_schema:
        _create_legacy_main_database(path)

    graph = _graph(path)
    try:
        columns = {
            str(row[1])
            for row in graph.connection.execute(
                "PRAGMA table_info(validation_records)"
            ).fetchall()
        }
        assert LEGACY_VALIDATION_COLUMNS < columns
        assert "assessment_id" in columns

        graph.add_variants((_variant(),))
        graph.add_validation_records((_validation_record(),))

        stored = graph.connection.execute(
            """
            SELECT reflection_id, reflection_verdict, reflection_summary,
                   reflection_quality_status, reflection_advisory_only, assessment_id
            FROM validation_records WHERE record_id = 'VR01-001'
            """
        ).fetchone()
        assert stored == (
            "r1",
            "support",
            "The measured value supports the sample rationale.",
            "model",
            1,
            "ha1",
        )

        prior = graph.validation_prior_context(round_id=2, limit=5)[0]
        assert prior["reflection_id"] == "r1"
        assert prior["reflection_verdict"] == "support"
        assert prior["reflection_quality_status"] == "model"
        assert prior["reflection_advisory_only"] is True
        assert prior["assessment_id"] == "ha1"

        history = graph.explain_variant("v1", round_id=2)["validation_history"][0]
        assert history["reflection_summary"].startswith("The measured value")
        assert history["assessment_id"] == "ha1"

        edge = next(
            item
            for item in graph.export_edges()
            if item.get("entity_id") == "VR01-001"
        )
        assert edge["reflection_id"] == "r1"
        assert edge["reflection_verdict"] == "support"
        assert edge["assessment_id"] == "ha1"
    finally:
        graph.close()


def test_engine_keeps_legacy_and_hypothesis_level_learning_in_parallel(
    experiment_config,
    tmp_path: Path,
) -> None:
    engine = KnowledgeEngine(
        experiment_config.knowledge,
        graph_path=tmp_path / "engine-kg.sqlite",
        structured_graph_path=tmp_path / "structured-kg.sqlite",
        assay_id="compatibility-assay",
        local_knowledge_enabled=False,
    )
    variant = Variant("v1", "VDGV", "VDGV", "WT", 0, "oracle_pool")
    validation = _validation_record()
    legacy_reflection = ReThinkReflection(
        reflection_id="r1",
        variant_id="v1",
        round_id=1,
        verdict="support",
        summary="The measured value supports the sample rationale.",
        positive_findings=("The wet value is favorable.",),
        negative_findings=(),
        revised_reason="Keep the claim bounded to this sample.",
        next_round_advice="Test a matched alternative.",
        provider="mock_rethink",
        assessment_id="ha1",
        assessment_status="SUPPORTED",
    )
    criterion = CriterionResult(
        criterion_id="c1",
        signal=CriterionSignal.SUPPORT,
        metric_value=0.9,
        comparator_value=0.5,
        effect_size=0.4,
        observation_ids=("v1",),
        qc_status="ok",
        detector_name="mean_delta",
        detector_version="v1",
        reason_code="target_above_control",
    )
    assessment = HypothesisAssessment(
        assessment_id="ha1",
        hypothesis_id="h1",
        falsification_spec_id="fs1",
        round_id=1,
        status=HypothesisStatus.SUPPORTED,
        criterion_results=(criterion,),
        observation_ids=("v1",),
        decisive_criterion_ids=("c1",),
        unresolved_criterion_ids=(),
        evaluator_version="v1",
    )
    hypothesis_reflection = HypothesisReflection(
        reflection_id="hr1",
        hypothesis_id="h1",
        assessment_id="ha1",
        round_id=1,
        assessment_status="SUPPORTED",
        summary="The hypothesis is retained within the measured scope.",
        retained_claims=("Retain the bounded claim.",),
        invalidated_assumptions=(),
        unresolved_questions=("Generalization remains unresolved.",),
        recommended_actions=("test_matched_alternative",),
        supporting_observation_ids=("v1",),
        supporting_evidence_ids=("e1",),
        provider="mock_rethink",
    )
    captured = {}

    class _CapturingBuilder:
        def build(self, context):
            captured["context"] = context
            return context

    try:
        engine.graph.add_variants((variant,))
        # The dev call shape (records only) and the legacy main call shape
        # (records plus reflections) must both remain valid after the merge.
        engine.record_validation((validation,))
        engine.record_validation((), (legacy_reflection,))
        engine.graph.add_hypothesis("h1", 1, "A bounded hypothesis.", ("e1",))
        engine.record_hypothesis_learning(assessment, hypothesis_reflection)

        assert engine._reflections == {"r1": legacy_reflection}
        assert engine._hypothesis_assessments == {"ha1": assessment}
        assert engine._hypothesis_reflections == {"hr1": hypothesis_reflection}
        assert engine.graph.connection.execute(
            "SELECT COUNT(*) FROM hypothesis_assessments"
        ).fetchone()[0] == 1
        assert engine.graph.connection.execute(
            "SELECT COUNT(*) FROM hypothesis_reflections"
        ).fetchone()[0] == 1

        engine.structured_builder = _CapturingBuilder()
        engine.sync_structured_kg(
            run_id="compatibility-run",
            round_id=2,
            variants=(variant,),
            observations=(),
            hypotheses=(),
        )
        resources = captured["context"].resources
        assert resources["reflections"] == (legacy_reflection,)
        assert resources["hypothesis_assessments"] == (assessment,)
        assert resources["hypothesis_reflections"] == (hypothesis_reflection,)
    finally:
        engine.close()
