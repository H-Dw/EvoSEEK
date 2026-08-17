import pytest

from fitness_agents.kg_interaction import (
    ChangeOperation,
    CompareVariantsOperator,
    EvidenceProvenanceOperator,
    EvidenceSufficiencyPolicy,
    ExplainVariantOperator,
    FeatureEvidenceOperator,
    HypothesisContextOperator,
    InMemoryChangeWriter,
    InteractionAblationConfig,
    KGChangeProposal,
    KGInteractionController,
    KGQueryContext,
    KGQueryPlan,
    KGQueryStep,
    ProposalGateway,
    QueryIntent,
    RoundScopedToolExecutor,
)
from fitness_agents.plugin_registry import PluginRegistry


class FakeGraphTool:
    def hypothesis_context(self, *, round_id, limit):
        return {
            "query_id": f"context:{round_id}",
            "beneficial_site_residues": [{"position": 3, "residue": "W"}],
            "current_candidate_predictions": [],
            "prior_hypotheses": [],
            "limit": limit,
        }

    def explain_variant(self, variant_id, *, round_id):
        score = -0.4 if variant_id == "v_bad" else 0.7
        return {
            "query_id": f"explain:{round_id}:{variant_id}",
            "variant_id": variant_id,
            "found": True,
            "mutation_notation": "A3W",
            "visible_observations": [],
            "predictions": [{"fitness_mean": 0.8}],
            "evidence": [
                {
                    "evidence_id": f"ev:{variant_id}",
                    "score": score,
                    "source_id": "model:test",
                    "evidence_type": "computed",
                }
            ],
        }

    def feature_evidence(self, variant_id, *, channel, round_id):
        return {
            "query_id": f"feature:{round_id}:{variant_id}:{channel}",
            "variant_id": variant_id,
            "channel": channel,
            "evidence": [
                {
                    "evidence_id": "ev:feature",
                    "channel": channel,
                    "score": 0.2,
                    "warnings": ["descriptor_only"],
                    "provenance": {"resource_sha256": "abc"},
                }
            ],
        }

    def evidence_provenance(self, evidence_id, *, round_id):
        return {
            "query_id": f"provenance:{round_id}:{evidence_id}",
            "evidence_id": evidence_id,
            "found": True,
            "warnings": ["descriptor_only"],
            "provenance": {"resource_sha256": "abc"},
        }


def _registry():
    tool = FakeGraphTool()
    registry = PluginRegistry("query_operator")
    registry.register("hypothesis_context", HypothesisContextOperator(tool))
    registry.register("explain_variant", ExplainVariantOperator(tool))
    registry.register("compare_variants", CompareVariantsOperator(tool))
    registry.register(
        "query_physchem_delta",
        FeatureEvidenceOperator("query_physchem_delta", "physchem", tool),
    )
    registry.register(
        "query_evidence_provenance", EvidenceProvenanceOperator(tool)
    )
    return registry


def test_controller_executes_bounded_plan_and_can_ablate_counterevidence():
    controller = KGInteractionController(
        _registry(),
        config=InteractionAblationConfig(
            max_tool_calls=3,
            use_counterevidence=False,
            stop_when_sufficient=False,
        ),
    )
    plan = KGQueryPlan(
        "plan:1",
        "rank variants",
        (
            KGQueryStep("s1", "hypothesis_context", QueryIntent.CONTEXT),
            KGQueryStep(
                "s2",
                "explain_variant",
                QueryIntent.COUNTEREVIDENCE,
                {"variant_id": "v_bad"},
                ("s1",),
            ),
            KGQueryStep(
                "s3",
                "compare_variants",
                QueryIntent.COMPARE,
                {"variant_ids": ["v_good", "v_bad"]},
                ("s1",),
            ),
        ),
    )
    result = controller.execute(
        plan,
        KGQueryContext("run:1", 2, frozenset({"v_good", "v_bad"})),
    )
    assert result.executed_steps == ("s1", "s3")
    assert ("s2", "counterevidence_ablation") in result.skipped_steps
    assert result.packs[-1].has_counterevidence


def test_controller_rejects_raw_query_and_out_of_scope_variant():
    controller = KGInteractionController(_registry())
    raw_plan = KGQueryPlan(
        "plan:raw",
        "unsafe",
        (KGQueryStep("s1", "hypothesis_context", QueryIntent.CONTEXT, {"sql": "SELECT 1"}),),
    )
    with pytest.raises(ValueError, match="forbidden"):
        controller.execute(raw_plan, KGQueryContext("run:1", 1))

    scope_plan = KGQueryPlan(
        "plan:scope",
        "unsafe scope",
        (KGQueryStep("s1", "explain_variant", QueryIntent.EXPLAIN, {"variant_id": "hidden"}),),
    )
    with pytest.raises(ValueError, match="out-of-scope"):
        controller.execute(scope_plan, KGQueryContext("run:1", 1, frozenset({"visible"})))


def test_sufficiency_and_proposal_gateway_are_independently_testable():
    controller = KGInteractionController(
        _registry(),
        config=InteractionAblationConfig(max_tool_calls=2, use_counterevidence=False),
        sufficiency=EvidenceSufficiencyPolicy(min_fact_count=1),
    )
    result = controller.execute(
        KGQueryPlan(
            "plan:stop",
            "get one fact",
            (
                KGQueryStep("s1", "hypothesis_context", QueryIntent.CONTEXT),
                KGQueryStep("s2", "hypothesis_context", QueryIntent.CONTEXT),
            ),
        ),
        KGQueryContext("run:1", 1),
    )
    assert result.executed_steps == ("s1",)
    assert result.stop_reason == "evidence_sufficient"

    proposal = KGChangeProposal(
        "proposal:1",
        "scientist",
        ChangeOperation.CHANGE_HYPOTHESIS_STATUS,
        "hypothesis:h1",
        {"status": "supported"},
        ("evidence:e1",),
        "run:1:h1:supported",
        0.8,
    )
    writer = InMemoryChangeWriter()
    assert ProposalGateway(writer).submit(proposal).status == "dry_run"
    gateway = ProposalGateway(writer, read_only=False)
    assert gateway.submit(proposal).status == "committed"
    assert gateway.submit(proposal).status == "duplicate"


def test_invalid_hypothesis_transition_is_rejected():
    proposal = KGChangeProposal(
        "proposal:bad",
        "scientist",
        ChangeOperation.CHANGE_HYPOTHESIS_STATUS,
        "hypothesis:h1",
        {"status": "certain"},
        ("evidence:e1",),
        "bad-status",
        1.0,
    )
    result = ProposalGateway(InMemoryChangeWriter(), read_only=False).submit(proposal)
    assert result.status == "rejected"
    assert "invalid hypothesis status" in result.errors


def test_round_scoped_tool_executor_records_calls_and_enforces_budget():
    session = RoundScopedToolExecutor(
        KGInteractionController(_registry()),
        KGQueryContext("run:sdk", 1, frozenset({"v_good", "v_bad"}), max_rows=4),
        plan_id="kgplan:sdk",
        max_tool_calls=1,
    )
    pack = session.call("hypothesis_context", QueryIntent.CONTEXT, {"limit": 4})
    assert pack["operator"] == "hypothesis_context"
    assert session.query_ids
    with pytest.raises(RuntimeError, match="budget"):
        session.call("explain_variant", QueryIntent.EXPLAIN, {"variant_id": "v_good"})
    result = session.result()
    assert result.stop_reason == "tool_call_budget_exhausted"
    assert result.executed_steps == ("tool_01",)


def test_feature_and_provenance_tools_remain_scoped_and_bounded():
    controller = KGInteractionController(
        _registry(),
        config=InteractionAblationConfig(
            enabled_operators=frozenset(
                {"query_physchem_delta", "query_evidence_provenance"}
            ),
            max_tool_calls=2,
            stop_when_sufficient=False,
        ),
    )
    result = controller.execute(
        KGQueryPlan(
            "plan:features",
            "inspect typed feature evidence",
            (
                KGQueryStep(
                    "feature",
                    "query_physchem_delta",
                    QueryIntent.EXPLAIN,
                    {"variant_id": "v_good"},
                ),
                KGQueryStep(
                    "provenance",
                    "query_evidence_provenance",
                    QueryIntent.HISTORY,
                    {"evidence_id": "ev:feature"},
                    ("feature",),
                ),
            ),
            max_tool_calls=2,
        ),
        KGQueryContext("run:1", 2, frozenset({"v_good"}), max_rows=2),
    )

    assert result.executed_steps == ("feature", "provenance")
    assert result.packs[0].evidence[0]["channel"] == "physchem"
    assert result.packs[1].provenance[0]["resource_sha256"] == "abc"
