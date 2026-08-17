from __future__ import annotations

import json
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
                "predictions and computed evidence remain explicitly typed and are not measurements; "
                "validation priors are append-only and retain wet/dry provenance"
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

    def feature_evidence(
        self,
        variant_id: str,
        *,
        channel: str,
        round_id: int,
    ) -> dict[str, Any]:
        if channel not in {"physchem", "conservation", "structure", "kg"}:
            raise ValueError(f"Unsupported feature evidence channel: {channel}")
        result = self.graph.explain_variant(variant_id, round_id=round_id)
        filtered = [
            item for item in result.get("evidence", ()) if item.get("channel") == channel
        ]
        payload = {
            "variant_id": variant_id,
            "found": bool(result.get("found", False)),
            "channel": channel,
            "evidence": filtered[: self.max_rows],
        }
        query_id = self.graph.record_agent_query(
            "feature_evidence",
            round_id=round_id,
            parameters={"variant_id": variant_id, "channel": channel},
            result=payload,
        )
        return {
            "tool": self.tool_name,
            "query_id": query_id,
            "operation": "feature_evidence",
            "as_of_round": round_id,
            **payload,
        }

    def evidence_provenance(self, evidence_id: str, *, round_id: int) -> dict[str, Any]:
        row = self.graph.connection.execute(
            """
            SELECT evidence_id, variant_id, channel, source_id, quality_status,
                   applicability, calibrated, warnings_json, provenance_json
            FROM evidence WHERE evidence_id = ? AND round_id <= ?
            """,
            (evidence_id, round_id),
        ).fetchone()
        result = (
            {"evidence_id": evidence_id, "found": False}
            if row is None
            else {
                "evidence_id": row[0],
                "variant_id": row[1],
                "channel": row[2],
                "source_id": row[3],
                "quality_status": row[4],
                "applicability": row[5],
                "calibrated": bool(row[6]),
                "warnings": json.loads(row[7]),
                "provenance": json.loads(row[8]),
                "found": True,
            }
        )
        query_id = self.graph.record_agent_query(
            "evidence_provenance",
            round_id=round_id,
            parameters={"evidence_id": evidence_id},
            result=result,
        )
        return {"tool": self.tool_name, "query_id": query_id, **result}
