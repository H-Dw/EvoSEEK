from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from fitness_agents.agents.researcher import MockResearcherClient
from fitness_agents.config import (
    LeakageGuardConfig,
    LocalKnowledgeConfig,
    LocalKnowledgeIngestionConfig,
    LocalKnowledgeRetrievalConfig,
    LocalKnowledgeRootConfig,
    ResearcherConfig,
)
from fitness_agents.contracts.researcher import (
    ExternalRetrievalPlan,
    FeatureEvidenceNeed,
    FeatureEvidencePlan,
    RetrievalNeed,
)
from fitness_agents.loop.orchestrator import CampaignRunner

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _agentic_local_config(tmp_path: Path) -> LocalKnowledgeConfig:
    return LocalKnowledgeConfig(
        enabled=True,
        corpus_index_path=tmp_path / "agentic-candidate.sqlite",
        retrieval_overlay_path=tmp_path / "agentic-overlay.sqlite",
        roots=(
            LocalKnowledgeRootConfig(
                path=PROJECT_ROOT / "resources/local_knowledge/gb1_agentic_candidate_v1",
                root_id="AGENTIC_INTEGRATION",
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
            top_k=2,
            token_budget=1600,
            max_chunks_per_document=1,
            instruction_content_policy="reject",
        ),
        leakage_guard=LeakageGuardConfig(enabled=False),
        allow_remote_context=True,
    )


def _agentic_runtime_config(experiment_config, tmp_path: Path, *, local) -> object:
    return replace(
        experiment_config,
        rounds=1,
        budget_per_round=2,
        candidate_limit=24,
        output_root=tmp_path / "runs",
        run_label="researcher-agentic",
        knowledge=replace(experiment_config.knowledge, local_knowledge=local),
        researcher=ResearcherConfig(enabled=True, provider="mock"),
        kg_interaction=replace(
            experiment_config.kg_interaction,
            feature_tool_strategy="agentic",
            feature_variant_limit=2,
            max_tool_calls=8,
        ),
    )


class _ContextAwareFeatureResearcher(MockResearcherClient):
    def plan_features(self, context):
        self.calls.append(context)
        sample_id = context.sample_map[0].sample_id
        return FeatureEvidencePlan(
            decision="PLAN",
            needs=(
                FeatureEvidenceNeed(
                    request_id="F1",
                    sample_id=sample_id,
                    channel="structure",
                    positions=(39,),
                    focus=("interaction_flags",),
                    information_need=(
                        "Check the named structural boundary for one visible sample."
                    ),
                    rationale=(
                        "Only interaction flags are required by the retrieved boundary record."
                    ),
                ),
            ),
        )


@pytest.mark.integration
def test_agentic_campaign_calls_two_phases_once_and_avoids_duplicate_rag_query(
    experiment_config,
    tmp_path: Path,
) -> None:
    external = ExternalRetrievalPlan(
        decision="PLAN",
        evidence_gap="The measurement does not define the limit of static proximity evidence.",
        needs=(
            RetrievalNeed(
                need_id="N1",
                intent="boundary",
                scientific_question=(
                    "When can matched assay observations and static contact geometry support "
                    "a bounded mechanism explanation?"
                ),
                facets={
                    "record_type": ("logic_unit",),
                    "evidence_role": ("boundary",),
                },
                top_k=2,
                rationale="A boundary record prevents static proximity from becoming a rank.",
            ),
        ),
    )
    researcher = _ContextAwareFeatureResearcher(
        external_plans=(external,),
    )
    config = _agentic_runtime_config(
        experiment_config,
        tmp_path,
        local=_agentic_local_config(tmp_path),
    )

    summary = CampaignRunner(config, researcher_client=researcher).run()
    run_dir = config.output_root / summary["run_id"]
    receipt = json.loads(
        (run_dir / "round_01/researcher_round_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    interaction = json.loads(
        (run_dir / "round_01/kg_interaction.json").read_text(encoding="utf-8")
    )
    assert len(researcher.calls) == 2
    assert [item.phase for item in researcher.calls] == [
        "external_retrieval",
        "feature_evidence",
    ]
    assert {item.assay.assay_id for item in researcher.calls} == {"A1"}
    assert researcher.calls[1].rag_records
    assert receipt["budget_used"]["rag_queries"] == 1
    assert len(receipt["query_ids"]) == 1
    operators = [item["operator"] for item in interaction["packs"]]
    assert operators.count("query_structure_environment") == 1
    assert "query_local_knowledge" not in operators
    assert "query_structured_claims" not in operators


@pytest.mark.integration
def test_no_rag_skips_phase_a_and_feature_abstain_executes_no_feature_tools(
    experiment_config,
    tmp_path: Path,
) -> None:
    researcher = MockResearcherClient(
        feature_plans=(
            FeatureEvidencePlan(
                decision="ABSTAIN",
                abstention_reason="No feature projection is needed for this fixture.",
            ),
        ),
    )
    config = _agentic_runtime_config(
        experiment_config,
        tmp_path,
        local=LocalKnowledgeConfig(enabled=False),
    )

    summary = CampaignRunner(config, researcher_client=researcher).run()
    run_dir = config.output_root / summary["run_id"]
    interaction = json.loads(
        (run_dir / "round_01/kg_interaction.json").read_text(encoding="utf-8")
    )

    assert len(researcher.calls) == 1
    assert researcher.calls[0].phase == "feature_evidence"
    operators = {item["operator"] for item in interaction["packs"]}
    assert operators.isdisjoint(
        {
            "query_physchem_delta",
            "query_evolutionary_profile",
            "query_structure_environment",
        }
    )


@pytest.mark.integration
def test_agentic_external_abstain_still_allows_dynamic_feature_plan(
    experiment_config,
    tmp_path: Path,
) -> None:
    researcher = _ContextAwareFeatureResearcher(
        external_plans=(
            ExternalRetrievalPlan(
                decision="ABSTAIN",
                evidence_gap="",
                abstention_reason="No novel external evidence gap is present.",
            ),
        ),
    )
    config = _agentic_runtime_config(
        experiment_config,
        tmp_path,
        local=_agentic_local_config(tmp_path),
    )

    summary = CampaignRunner(config, researcher_client=researcher).run()
    run_dir = config.output_root / summary["run_id"]
    receipt = json.loads(
        (run_dir / "round_01/researcher_round_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    interaction = json.loads(
        (run_dir / "round_01/kg_interaction.json").read_text(encoding="utf-8")
    )
    candidate_pool = json.loads(
        (run_dir / "round_01/candidate_pool_receipt.json").read_text(
            encoding="utf-8"
        )
    )

    assert [item.phase for item in researcher.calls] == [
        "external_retrieval",
        "feature_evidence",
    ]
    assert receipt["external_plan"]["decision"] == "ABSTAIN"
    assert receipt["budget_used"]["rag_queries"] == 0
    assert receipt["feature_plan"]["decision"] == "PLAN"
    assert any(
        item["operator"] == "query_structure_environment"
        for item in interaction["packs"]
    )
    assert sum(candidate_pool["candidate_mutation_order_counts"].values()) == 24
    assert (
        candidate_pool["hypothesis_match_policy"]
        == "edited_non_wild_type_sites_only"
    )
    assert candidate_pool["evidence_prefilter_policy"] == "selection_authorized_only"


class _FailingResearcher(MockResearcherClient):
    def plan_external(self, context):
        self.calls.append(context)
        raise RuntimeError("synthetic remote failure")


@pytest.mark.integration
def test_researcher_failure_is_audited_and_aborts_round_without_fallback(
    experiment_config,
    tmp_path: Path,
) -> None:
    researcher = _FailingResearcher()
    config = _agentic_runtime_config(
        experiment_config,
        tmp_path,
        local=_agentic_local_config(tmp_path),
    )

    summary = CampaignRunner(config, researcher_client=researcher).run()
    run_dir = config.output_root / summary["run_id"]
    failure = json.loads(
        (run_dir / "round_01/researcher_failure.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (run_dir / "round_01/researcher_round_receipt.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["rounds_aborted"] == 1
    assert summary["queries_used"] == 0
    assert failure["failure_policy"] == "abort_round"
    assert receipt["rejected"][0]["step_id"] == "external_retrieval"
    assert not (run_dir / "round_01/local_rag_retrieval.json").exists()
