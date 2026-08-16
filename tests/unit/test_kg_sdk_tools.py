from __future__ import annotations

import pytest

from fitness_agents.kg_interaction import (
    CallableQueryOperator,
    EvidencePack,
    KGInteractionController,
    KGQueryContext,
    KGToolSession,
    QueryIntent,
)
from fitness_agents.plugin_registry import PluginRegistry


def _controller(*, rows: int = 1) -> KGInteractionController:
    registry = PluginRegistry("query_operator")

    def execute(step, context):
        variant_id = str(step.arguments.get("variant_id", "v1"))
        return EvidencePack(
            query_id=f"kgq:{variant_id}",
            operator="explain_variant",
            as_of_round=context.round_id,
            facts=tuple({"variant_id": variant_id, "row": index} for index in range(rows)),
        )

    registry.register(
        "explain_variant", CallableQueryOperator("explain_variant", execute)
    )
    return KGInteractionController(registry)


def _session(
    *, rows: int = 1, max_rows: int = 2, max_tool_calls: int = 1
) -> KGToolSession:
    return KGToolSession(
        _controller(rows=rows),
        KGQueryContext(
            run_id="run",
            round_id=2,
            allowed_variant_ids=frozenset({"v1"}),
            max_rows=max_rows,
        ),
        plan_id="kgplan:run:r2:sdk",
        max_tool_calls=max_tool_calls,
    )


def test_sdk_kg_session_enforces_scope_and_global_query_budget() -> None:
    session = _session(max_tool_calls=2)

    with pytest.raises(ValueError, match="out-of-scope"):
        session.call("explain_variant", QueryIntent.EXPLAIN, {"variant_id": "v2"})
    result = session.call(
        "explain_variant", QueryIntent.EXPLAIN, {"variant_id": "v1"}
    )
    with pytest.raises(RuntimeError, match="budget exhausted"):
        session.call("explain_variant", QueryIntent.EXPLAIN, {"variant_id": "v1"})

    assert result["as_of_round"] == 2
    assert session.query_ids == ("kgq:v1",)
    assert session.result().stop_reason == "tool_call_budget_exhausted"
    assert session.result().skipped_steps == (("sdk_tool_01", "rejected:ValueError"),)


def test_sdk_kg_session_rejects_operator_rows_over_round_limit() -> None:
    session = _session(rows=3, max_rows=2)

    with pytest.raises(ValueError, match="more than 2 rows"):
        session.call("explain_variant", QueryIntent.EXPLAIN, {"variant_id": "v1"})
