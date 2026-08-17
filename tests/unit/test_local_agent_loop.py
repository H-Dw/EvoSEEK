from __future__ import annotations

from fitness_agents.agents.local_agent_loop import LocalAgentLoop
from fitness_agents.agents.tool_contracts import AgentStep, ToolCall, ToolSpec
from fitness_agents.kg_interaction import (
    CallableQueryOperator,
    EvidencePack,
    KGInteractionController,
    KGQueryContext,
    QueryIntent,
    RoundScopedToolExecutor,
)
from fitness_agents.plugin_registry import PluginRegistry


def test_local_agent_loop_uses_only_allowlisted_round_scoped_tool() -> None:
    registry = PluginRegistry("query_operator")
    registry.register(
        "explain_variant",
        CallableQueryOperator(
            "explain_variant",
            lambda step, context: EvidencePack(
                query_id="kgq:v1",
                operator="explain_variant",
                as_of_round=context.round_id,
                facts=({"variant_id": step.arguments["variant_id"]},),
            ),
        ),
    )
    executor = RoundScopedToolExecutor(
        KGInteractionController(registry),
        KGQueryContext(
            run_id="run", round_id=1,
            allowed_variant_ids=frozenset({"v1"}), max_rows=2,
        ),
        plan_id="plan", max_tool_calls=1,
    )
    steps = iter(
        [
            AgentStep(tool_call=ToolCall(name="explain_variant", arguments={"variant_id": "v1"})),
            AgentStep(final_output={"hypothesis_id": "hyp:run:r1"}),
        ]
    )
    output = LocalAgentLoop(max_turns=2).run(
        next_step=lambda messages, tools: next(steps),
        messages=({"role": "user", "content": "visible context"},),
        tools=(ToolSpec(
            name="explain_variant", description="read-only explanation",
            intent=QueryIntent.EXPLAIN, input_schema={"type": "object"},
        ),),
        executor=executor,
    )
    assert output == {"hypothesis_id": "hyp:run:r1"}
    assert executor.query_ids == ("kgq:v1",)
