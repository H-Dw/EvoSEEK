from __future__ import annotations

from typing import Any

from .graph import ObservationKnowledgeGraph


class AgentKnowledgeGraphTool:
    """Bounded, allow-listed knowledge-graph interface for scientist agents.

    The tool exposes typed query operations rather than raw SQL. All query results are persisted in
    the graph for replay and audit, and the graph itself enforces round-based observation visibility.
    """

    tool_name = "knowledge_graph"

    def __init__(self, graph: ObservationKnowledgeGraph, *, max_rows: int = 12) -> None:
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        self.graph = graph
        self.max_rows = max_rows

    def hypothesis_context(self, *, round_id: int, limit: int | None = None) -> dict[str, Any]:
        effective_limit = min(max(1, limit or self.max_rows), self.max_rows)
        result = self.graph.agent_hypothesis_context(
            round_id=round_id,
            limit=effective_limit,
        )
        parameters = {"limit": effective_limit}
        query_id = self.graph.record_agent_query(
            "hypothesis_context",
            round_id=round_id,
            parameters=parameters,
            result=result,
        )
        return {
            "tool": self.tool_name,
            "query_id": query_id,
            "operation": "hypothesis_context",
            "as_of_round": round_id,
            "visibility_rule": (
                "measurements require round_revealed < as_of_round; current-round model "
                "predictions and computed evidence remain explicitly typed and are not measurements"
            ),
            **result,
        }

    def explain_variant(self, variant_id: str, *, round_id: int) -> dict[str, Any]:
        result = self.graph.explain_variant(variant_id, round_id=round_id)
        parameters = {"variant_id": variant_id}
        query_id = self.graph.record_agent_query(
            "explain_variant",
            round_id=round_id,
            parameters=parameters,
            result=result,
        )
        return {
            "tool": self.tool_name,
            "query_id": query_id,
            "operation": "explain_variant",
            "as_of_round": round_id,
            **result,
        }
