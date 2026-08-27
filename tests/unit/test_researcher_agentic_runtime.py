from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fitness_agents.agents.researcher import MockResearcherClient, NativeResearcherClient
from fitness_agents.config import (
    LeakageGuardConfig,
    LocalKnowledgeConfig,
    LocalKnowledgeIngestionConfig,
    LocalKnowledgeKGUpdateConfig,
    LocalKnowledgeRetrievalConfig,
    LocalKnowledgeRootConfig,
    ResearcherConfig,
    load_experiment_config,
)
from fitness_agents.contracts.researcher import (
    ExternalRetrievalPlan,
    FeatureEvidenceNeed,
    FeatureEvidencePlan,
    ResearcherAssayContext,
    ResearcherContextInput,
    ResearcherFacetCatalog,
    ResearcherKnowledgeRecordCard,
    ResearcherSampleCard,
    ResearcherToolCard,
    RetrievalNeed,
)
from fitness_agents.kg_interaction.researcher import ResearcherPlanningController
from fitness_agents.knowledge.tool import AgentKnowledgeGraphTool
from fitness_agents.local_knowledge.index import SQLiteLocalKnowledgeIndex

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _QueuedResearcherTransport:
    base_url = "https://api.deepseek.com"

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        content = json.dumps(self.payloads.pop(0))
        message = type("Message", (), {"content": content, "reasoning_content": ""})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type(
            "Response",
            (),
            {"choices": [choice], "usage": None, "model": "deepseek-v4-pro"},
        )()


def _assay() -> ResearcherAssayContext:
    return ResearcherAssayContext(
        assay_id="binding-assay",
        objective="Improve measured association while preserving a credible fold boundary.",
        fitness_scale="relative measured response",
        optimization_direction="higher_is_better",
    )


def _sample(sample_id: str = "S1") -> ResearcherSampleCard:
    return ResearcherSampleCard(
        sample_id=sample_id,
        observation_id=f"OBS-{sample_id}",
        measured_fitness=0.0,
        round_revealed=0,
        source="wet_experiment",
        mutated_positions=(),
    )


def _facets() -> ResearcherFacetCatalog:
    return ResearcherFacetCatalog(
        allowed_values={
            "record_type": ("atomic_claim", "logic_unit", "knowledge_decision_card"),
            "feature_channel": ("physchem", "conservation", "structure"),
            "evidence_role": ("support", "boundary", "counterevidence"),
        }
    )


def test_support_plan_requires_counterevidence_or_boundary() -> None:
    with pytest.raises(ValidationError, match="paired"):
        ExternalRetrievalPlan(
            decision="PLAN",
            evidence_gap="A mechanism needs an external constraint.",
            needs=(
                RetrievalNeed(
                    need_id="N1",
                    intent="support",
                    scientific_question=(
                        "Which physical observations support the proposed mechanism in a "
                        "non-pathogen protein binding assay?"
                    ),
                    rationale="The proposed mechanism lacks direct external support.",
                ),
            ),
        )


def test_controller_fails_closed_on_unknown_facets_and_samples() -> None:
    config = ResearcherConfig(enabled=True)
    controller = ResearcherPlanningController(
        config,
        mutable_positions=(39, 40, 41, 54),
        facet_catalog=_facets().allowed_values,
        forbidden_query_terms=("protected-task",),
    )
    invalid_external = ExternalRetrievalPlan(
        decision="PLAN",
        evidence_gap="The available observation does not resolve assay transfer.",
        needs=(
            RetrievalNeed(
                need_id="N1",
                intent="boundary",
                scientific_question=(
                    "Under which assay boundaries can static structural proximity support "
                    "a mechanistic explanation?"
                ),
                facets={"feature_channel": ("not_a_channel",)},
                rationale="A boundary record is needed to avoid overclaiming proximity.",
            ),
        ),
    )
    with pytest.raises(ValueError, match="out-of-catalog"):
        controller.validate_external_plan(invalid_external)

    invalid_feature = FeatureEvidencePlan(
        decision="PLAN",
        needs=(
            FeatureEvidenceNeed(
                request_id="F1",
                sample_id="S2",
                channel="structure",
                positions=(39,),
                focus=("contact_geometry",),
                information_need="Determine whether the visible site has local contacts.",
                rationale="A contact projection tests the stated structural boundary.",
            ),
        ),
    )
    with pytest.raises(ValueError, match="unknown sample"):
        controller.validate_feature_plan(
            invalid_feature,
            sample_id_to_variant_id={"S1": "internal-wt"},
        )


def test_query_language_scope_and_budgets_fail_closed() -> None:
    with pytest.raises(ValidationError, match="English"):
        RetrievalNeed(
            need_id="N1",
            intent="boundary",
            scientific_question="这个科学问题不是英文，因此不能进入检索执行阶段。",
            rationale="The request must fail before retrieval execution begins.",
        )
    controller = ResearcherPlanningController(
        ResearcherConfig(
            enabled=True,
            max_rag_queries=1,
            rag_top_k_per_query=1,
        ),
        mutable_positions=(39, 40, 41, 54),
        facet_catalog=_facets().allowed_values,
        forbidden_query_terms=("protected-task",),
    )
    over_budget = ExternalRetrievalPlan(
        decision="PLAN",
        evidence_gap="A single bounded request is sufficient for this fixture.",
        needs=(
            RetrievalNeed(
                need_id="N1",
                intent="boundary",
                scientific_question=(
                    "Which evidence boundary constrains a generic binding mechanism claim?"
                ),
                top_k=2,
                rationale="The fixture deliberately exceeds the configured top-k budget.",
            ),
        ),
    )
    with pytest.raises(ValueError, match="top_k budget"):
        controller.validate_external_plan(over_budget)
    for prohibited_query in (
        "Which viral protein evidence supports this generic binding mechanism?",
        "Which boundary supports the protected-task mechanism claim?",
        "Which boundary applies to the A39V mechanism claim?",
    ):
        plan = ExternalRetrievalPlan(
            decision="PLAN",
            evidence_gap="The query deliberately violates a controller boundary.",
            needs=(
                RetrievalNeed(
                    need_id="N1",
                    intent="boundary",
                    scientific_question=prohibited_query,
                    top_k=1,
                    rationale="The controller must reject this request without retrieval.",
                ),
            ),
        )
        with pytest.raises(ValueError):
            controller.validate_external_plan(plan)


def test_native_researcher_repairs_protected_identity_inside_semantic_retry() -> None:
    def payload(question: str) -> dict:
        return {
            "schema_version": "external-retrieval-plan:v1",
            "decision": "PLAN",
            "evidence_gap": "A fold-versus-binding boundary is not yet available.",
            "needs": [
                {
                    "need_id": "N1",
                    "intent": "boundary",
                    "scientific_question": question,
                    "facets": {},
                    "top_k": 1,
                    "rationale": "A boundary record prevents an unsupported mechanism claim.",
                }
            ],
            "abstention_reason": None,
        }

    transport = _QueuedResearcherTransport(
        [
            payload("Which boundary applies to the protected-task binding assay?"),
            payload(
                "Which evidence boundary separates folding failure from direct binding "
                "failure in the current assay?"
            ),
        ]
    )
    controller = ResearcherPlanningController(
        ResearcherConfig(enabled=True, max_rag_queries=1, rag_top_k_per_query=1),
        mutable_positions=(39, 40, 41, 54),
        facet_catalog=_facets().allowed_values,
        forbidden_query_terms=("protected-task",),
    )
    client = NativeResearcherClient(
        model="deepseek-v4-pro",
        provider="deepseek",
        transport=transport,
        thinking="disabled",
        max_transport_retries=0,
        max_truncation_retries=0,
        max_syntax_retries=0,
        max_schema_retries=0,
        max_semantic_retries=1,
    )
    client.bind_external_plan_validator(controller.validate_external_plan)
    context = ResearcherContextInput(
        phase="external_retrieval",
        run_id="RUN1",
        round_id=1,
        task="Improve the measured response.",
        assay=ResearcherAssayContext(
            assay_id="A1",
            objective="Improve the measured response.",
            fitness_scale="relative response",
            optimization_direction="higher_is_better",
        ),
        measurement_kg=(_sample(),),
        facet_catalog=_facets(),
    )

    plan = client.plan_external(context)

    assert len(transport.calls) == 2
    assert plan.needs[0].scientific_question.endswith("in the current assay?")
    assert "identity-neutral" in transport.calls[1]["messages"][-1]["content"]


def test_invalid_focus_and_position_never_become_tool_steps() -> None:
    with pytest.raises(ValidationError, match="does not belong"):
        FeatureEvidenceNeed(
            request_id="F1",
            sample_id="S1",
            channel="physchem",
            positions=(39,),
            focus=("contact_geometry",),
            information_need="Attempt to request a focus from the wrong feature channel.",
            rationale="The typed contract must reject cross-channel projection fields.",
        )
    controller = ResearcherPlanningController(
        ResearcherConfig(enabled=True),
        mutable_positions=(39, 40, 41, 54),
        facet_catalog=_facets().allowed_values,
    )
    outside = FeatureEvidencePlan(
        decision="PLAN",
        needs=(
            FeatureEvidenceNeed(
                request_id="F1",
                sample_id="S1",
                channel="structure",
                positions=(1,),
                focus=("contact_geometry",),
                information_need="Attempt to query a position outside the task scope.",
                rationale="The controller must reject positions not listed by the task.",
            ),
        ),
    )
    with pytest.raises(ValueError, match="outside task scope"):
        controller.validate_feature_plan(
            outside,
            sample_id_to_variant_id={"S1": "internal-wt"},
        )


def test_feature_plan_maps_short_id_and_projection_only() -> None:
    controller = ResearcherPlanningController(
        ResearcherConfig(enabled=True),
        mutable_positions=(39, 40, 41, 54),
        facet_catalog=_facets().allowed_values,
    )
    plan = FeatureEvidencePlan(
        decision="PLAN",
        needs=(
            FeatureEvidenceNeed(
                request_id="F1",
                sample_id="S1",
                channel="physchem",
                positions=(39,),
                focus=("special_flags",),
                information_need="Check whether a visible sample crosses a fold warning.",
                rationale="Only the named warning projection is necessary for this boundary.",
            ),
        ),
    )
    steps = controller.validate_feature_plan(
        plan,
        sample_id_to_variant_id={"S1": "internal-wt"},
    )
    assert len(steps) == 1
    assert steps[0].operator == "query_physchem_delta"
    assert steps[0].arguments == {
        "variant_id": "internal-wt",
        "positions": [39],
        "projection": ["special_flags"],
    }


def test_projection_keeps_quality_warnings_applicability_and_provenance() -> None:
    evidence = {
        "evidence_id": "E1",
        "round_id": 1,
        "channel": "structure",
        "statement": "Static structure descriptor",
        "quality_status": "degraded",
        "applicability": "boundary_only",
        "warnings": ["side chain unavailable"],
        "provenance": {"resource": "static-model"},
        "raw_features": {
            "sites": {
                "39": {
                    "mutation": "masked",
                    "contact_count": 4,
                    "closest_contacts": ["masked-neighbour"],
                    "relative_sasa": 0.25,
                },
                "40": {"mutation": "masked", "contact_count": 2},
            },
            "unrequested_secret": "must-not-be-returned",
        },
    }
    projected = AgentKnowledgeGraphTool._project_feature_evidence(
        evidence,
        channel="structure",
        projection=("contact_geometry",),
        positions=(39,),
    )
    assert set(projected["raw_features"]) == {"contact_geometry"}
    assert set(projected["raw_features"]["contact_geometry"]) == {"39"}
    assert projected["quality_status"] == "degraded"
    assert projected["warnings"] == ["side chain unavailable"]
    assert projected["applicability"] == "boundary_only"
    assert projected["provenance"] == {"resource": "static-model"}


def test_mock_researcher_receives_phase_b_native_record_ids() -> None:
    external = ExternalRetrievalPlan(
        decision="ABSTAIN",
        evidence_gap="",
        abstention_reason="The fixture has no external evidence gap.",
    )
    features = FeatureEvidencePlan(
        decision="ABSTAIN",
        abstention_reason="No feature request is needed for the fixture.",
    )
    client = MockResearcherClient(
        external_plans=(external,),
        feature_plans=(features,),
    )
    phase_a = ResearcherContextInput(
        phase="external_retrieval",
        run_id="run-1",
        round_id=1,
        task="Generic non-pathogen protein binding analysis",
        assay=_assay(),
        measurement_kg=(_sample(),),
        facet_catalog=_facets(),
    )
    assert client.plan_external(phase_a).decision == "ABSTAIN"
    record = ResearcherKnowledgeRecordCard(
        record_id="LU-1",
        record_type="logic_unit",
        retrieval_text="Use matched observations and abstain when assay conditions differ.",
        knowledge_type="matched_observation",
        permission="explanation_only",
        boundary_conditions=("Assay conditions must match.",),
        abstain_if=("No matched observation exists.",),
        facets={"evidence_role": ("boundary",)},
    )
    tool = ResearcherToolCard(
        tool_id="query_structure_environment",
        channel="structure",
        allowed_positions=(39, 40, 41, 54),
        allowed_focus=("contact_geometry",),
    )
    phase_b = ResearcherContextInput(
        phase="feature_evidence",
        run_id="run-1",
        round_id=1,
        task="Generic non-pathogen protein binding analysis",
        assay=_assay(),
        measurement_kg=(_sample(),),
        sample_map=(_sample(),),
        rag_records=(record,),
        facet_catalog=_facets(),
        tool_catalog=(tool,),
    )
    assert client.plan_features(phase_b).decision == "ABSTAIN"
    assert len(client.calls) == 2
    assert [item.record_id for item in client.calls[1].rag_records] == ["LU-1"]


def test_native_candidate_index_has_typed_records_and_exact_facets(tmp_path: Path) -> None:
    config = LocalKnowledgeConfig(
        enabled=True,
        corpus_index_path=tmp_path / "candidate.sqlite",
        roots=(
            LocalKnowledgeRootConfig(
                path=PROJECT_ROOT / "resources/local_knowledge/gb1_agentic_candidate_v1",
                root_id="AGENTIC_TEST",
                access_policy_mode="synthetic_test",
                runtime_manifest_mode="legacy_compatible",
                include=("records/**/*.md",),
            ),
        ),
        ingestion=LocalKnowledgeIngestionConfig(
            required_language="en",
            chunk_tokens=384,
            chunk_overlap=32,
        ),
        retrieval=LocalKnowledgeRetrievalConfig(
            query_mode="agentic",
            mode="lexical",
            instruction_content_policy="reject",
        ),
        kg_update=LocalKnowledgeKGUpdateConfig(
            source_group="agentic_candidate_test",
        ),
        leakage_guard=LeakageGuardConfig(enabled=False),
    )
    index = SQLiteLocalKnowledgeIndex(config.corpus_index_path)
    try:
        report = index.build(config)
        assert report.indexed_documents == 6
        catalog = index.facet_catalog()
        assert set(catalog["record_type"]) == {
            "atomic_claim",
            "logic_unit",
            "knowledge_decision_card",
        }
        hits = index.lexical_search(
            "matched observations pairwise signal",
            limit=10,
            facets={
                "record_type": ("logic_unit",),
                "feature_channel": ("conservation",),
            },
        )
        assert hits
        chunks = index.get_chunks(tuple(chunk_id for chunk_id, _ in hits))
        assert all(
            item["facets"]["record_type"] == ("logic_unit",)
            for item in chunks.values()
        )
    finally:
        index.close()


def test_agentic_config_keeps_existing_validation_budget_and_deepseek_model() -> None:
    config = load_experiment_config(
        PROJECT_ROOT
        / "configs/experiments/gb1_3features_agentic_researcher_deepseek_v4_pro.yaml"
    )
    assert (config.budget_per_round, config.candidate_limit, config.rounds) == (16, 32, 3)
    assert config.researcher.enabled is True
    assert config.researcher.model == "deepseek-v4-pro"
    assert config.knowledge.local_knowledge.retrieval.query_mode == "agentic"
    assert config.kg_interaction.feature_tool_strategy == "agentic"
