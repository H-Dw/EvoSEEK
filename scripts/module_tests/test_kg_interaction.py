from __future__ import annotations

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
from fitness_agents.kg_interaction import (
    ChangeOperation,
    CompareVariantsOperator,
    EvidenceSufficiencyPolicy,
    ExplainVariantOperator,
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
)
from fitness_agents.knowledge import KnowledgeEngine
from fitness_agents.plugin_registry import PluginRegistry


def main() -> None:
    args = parse_args("configs/module_tests/kg_interaction.yaml")
    config = load_config(args.config)
    output = resolve_output(config, args.output_dir)
    variants = variant_grid()
    observed = variants[:16]
    candidates = [variants[16], variants[40], variants[-1]]
    round_id = int(config["round_id"])
    engine = KnowledgeEngine(
        KnowledgeConfig(),
        graph_path=output / "interaction.sqlite",
        assay_id="module_test_assay",
    )
    engine.update(observed, make_observations(observed, round_revealed=0))
    evidence = engine.evidence_for(candidates, round_id=round_id)
    engine.record_inference_context(
        candidates,
        make_predictions(candidates, model_version="kg-interaction:v1"),
        evidence,
        round_id=round_id,
    )
    tool = engine.agent_tool(max_rows=int(config["max_rows"]))
    registry = PluginRegistry("query_operator")
    registry.register("hypothesis_context", HypothesisContextOperator(tool))
    registry.register("explain_variant", ExplainVariantOperator(tool))
    registry.register("compare_variants", CompareVariantsOperator(tool))

    interaction = InteractionAblationConfig.from_mapping(config["interaction"])
    sufficiency = EvidenceSufficiencyPolicy(**config["sufficiency"])
    controller = KGInteractionController(registry, config=interaction, sufficiency=sufficiency)
    allowed = frozenset(item.variant_id for item in candidates)
    plan = KGQueryPlan(
        "plan:module-test",
        "Collect context, candidate support, counterevidence, and a comparison.",
        (
            KGQueryStep("s1", "hypothesis_context", QueryIntent.CONTEXT),
            KGQueryStep(
                "s2",
                "explain_variant",
                QueryIntent.EXPLAIN,
                {"variant_id": candidates[0].variant_id},
                ("s1",),
            ),
            KGQueryStep(
                "s3",
                "explain_variant",
                QueryIntent.COUNTEREVIDENCE,
                {"variant_id": candidates[-1].variant_id},
                ("s1",),
            ),
            KGQueryStep(
                "s4",
                "compare_variants",
                QueryIntent.COMPARE,
                {"variant_ids": [candidates[0].variant_id, candidates[-1].variant_id]},
                ("s2", "s3"),
            ),
        ),
        max_tool_calls=4,
    )
    result = controller.execute(
        plan,
        KGQueryContext(str(config["run_id"]), round_id, allowed, int(config["max_rows"])),
    )
    ensure(result.executed_steps == ("s1", "s2", "s3", "s4"), "Query plan was not completed")
    ensure(any(pack.has_counterevidence for pack in result.packs), "Counterevidence was not retained")
    audited_queries = engine.graph.export_agent_queries()
    ensure(
        len(audited_queries) == 3
        and {item["operation"] for item in audited_queries}
        == {"hypothesis_context", "explain_variant"},
        "Unique underlying KG queries were not audited with idempotent query IDs",
    )

    unsafe_guard = False
    try:
        controller.execute(
            KGQueryPlan(
                "plan:unsafe",
                "Attempt raw query",
                (KGQueryStep("u1", "hypothesis_context", QueryIntent.CONTEXT, {"sql": "SELECT 1"}),),
            ),
            KGQueryContext(str(config["run_id"]), round_id, allowed),
        )
    except ValueError as error:
        unsafe_guard = "forbidden" in str(error)
    ensure(unsafe_guard, "Raw-query trust boundary did not reject SQL")

    scope_guard = False
    try:
        controller.execute(
            KGQueryPlan(
                "plan:scope",
                "Attempt out-of-scope lookup",
                (
                    KGQueryStep(
                        "o1",
                        "explain_variant",
                        QueryIntent.EXPLAIN,
                        {"variant_id": "hidden:variant"},
                    ),
                ),
            ),
            KGQueryContext(str(config["run_id"]), round_id, allowed),
        )
    except ValueError as error:
        scope_guard = "out-of-scope" in str(error)
    ensure(scope_guard, "Out-of-scope variant guard did not fire")

    proposal = KGChangeProposal(
        "proposal:module-test",
        "scientist",
        ChangeOperation.CHANGE_HYPOTHESIS_STATUS,
        "hypothesis:h1",
        {"status": "supported"},
        (next(iter(evidence.values()))[0].evidence_id,),
        "module-test:h1:supported",
        0.8,
    )
    writer = InMemoryChangeWriter()
    dry_run = ProposalGateway(writer).submit(proposal)
    gateway = ProposalGateway(writer, read_only=False)
    committed = gateway.submit(proposal)
    duplicate = gateway.submit(proposal)
    invalid = gateway.submit(
        KGChangeProposal(
            "proposal:invalid",
            "scientist",
            ChangeOperation.CHANGE_HYPOTHESIS_STATUS,
            "hypothesis:h1",
            {"status": "certain"},
            proposal.evidence_ids,
            "module-test:h1:invalid",
            1.0,
        )
    )
    ensure(
        (dry_run.status, committed.status, duplicate.status, invalid.status)
        == ("dry_run", "committed", "duplicate", "rejected"),
        "Proposal gateway state transitions are incorrect",
    )
    engine.close()

    write_result(
        output,
        "kg_interaction",
        {
            "config": config["_config_path"],
            "executed_steps": result.executed_steps,
            "stop_reason": result.stop_reason,
            "pack_fact_counts": [pack.fact_count for pack in result.packs],
            "counterevidence_packs": sum(pack.has_counterevidence for pack in result.packs),
            "unique_audited_queries": len(audited_queries),
            "trust_boundary": {"raw_query_rejected": unsafe_guard, "scope_rejected": scope_guard},
            "writeback_statuses": [dry_run.status, committed.status, duplicate.status, invalid.status],
        },
    )


if __name__ == "__main__":
    main()
