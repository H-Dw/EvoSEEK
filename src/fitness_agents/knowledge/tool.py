from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from fitness_agents.contracts.researcher import FEATURE_FOCUS_BY_CHANNEL

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
        projection: Sequence[str] = (),
        positions: Sequence[int] = (),
    ) -> dict[str, Any]:
        if channel not in {"physchem", "conservation", "structure", "kg"}:
            raise ValueError(f"Unsupported feature evidence channel: {channel}")
        result = self.graph.explain_variant(variant_id, round_id=round_id)
        if projection and channel not in FEATURE_FOCUS_BY_CHANNEL:
            raise ValueError(f"Feature projection is unsupported for channel: {channel}")
        requested_projection = tuple(dict.fromkeys(str(item) for item in projection))
        if not set(requested_projection).issubset(FEATURE_FOCUS_BY_CHANNEL.get(channel, ())):
            raise ValueError(f"Unsupported {channel} feature projection")
        requested_positions = tuple(dict.fromkeys(int(item) for item in positions))
        filtered = []
        for item in result.get("evidence", ()):
            if item.get("channel") != channel:
                continue
            if requested_projection:
                item = self._project_feature_evidence(
                    item,
                    channel=channel,
                    projection=requested_projection,
                    positions=requested_positions,
                )
            filtered.append(item)
        payload = {
            "variant_id": variant_id,
            "found": bool(result.get("found", False)),
            "channel": channel,
            "projection": list(requested_projection),
            "positions": list(requested_positions),
            "evidence": filtered[: self.max_rows],
        }
        query_id = self.graph.record_agent_query(
            "feature_evidence",
            round_id=round_id,
            parameters={
                "variant_id": variant_id,
                "channel": channel,
                "projection": list(requested_projection),
                "positions": list(requested_positions),
            },
            result=payload,
        )
        return {
            "tool": self.tool_name,
            "query_id": query_id,
            "operation": "feature_evidence",
            "as_of_round": round_id,
            **payload,
        }

    @staticmethod
    def _project_feature_evidence(
        evidence: Mapping[str, Any],
        *,
        channel: str,
        projection: tuple[str, ...],
        positions: tuple[int, ...],
    ) -> dict[str, Any]:
        """Return only allow-listed raw fields while retaining every limitation field."""

        raw = evidence.get("raw_features", {})
        raw = raw if isinstance(raw, Mapping) else {}
        position_keys = {str(item) for item in positions}

        def selected_sites() -> dict[str, Any]:
            sites = raw.get("sites", {})
            if not isinstance(sites, Mapping):
                return {}
            return {
                str(key): dict(value) if isinstance(value, Mapping) else value
                for key, value in sites.items()
                if not position_keys or str(key) in position_keys
            }

        sites = selected_sites()
        projected: dict[str, Any] = {}
        for focus in projection:
            if channel == "physchem":
                if focus == "site_deltas":
                    projected[focus] = {
                        key: {
                            name: value[name]
                            for name in ("mutation", "deltas")
                            if name in value
                        }
                        for key, value in sites.items()
                        if isinstance(value, Mapping)
                    }
                elif focus in {"global_sequence_deltas", "special_flags"}:
                    projected[focus] = raw.get(focus, {} if focus.endswith("deltas") else [])
            elif channel == "conservation":
                if focus == "site_log_odds":
                    projected[focus] = {
                        key: {
                            name: value[name]
                            for name in (
                                "mutation",
                                "wild_type_frequency",
                                "mutant_frequency",
                                "log_odds_vs_wild_type",
                                "effective_count",
                            )
                            if name in value
                        }
                        for key, value in sites.items()
                        if isinstance(value, Mapping)
                    }
                elif focus == "pairwise_signal":
                    projected[focus] = {
                        key: raw.get(key)
                        for key in (
                            "pairwise_frequency_log_odds",
                            "pairwise_residual_log_odds",
                            "pairwise_enabled",
                            "pairwise_eligible",
                            "pairwise_score_method",
                        )
                    }
                elif focus == "profile_quality":
                    projected[focus] = {
                        key: raw.get(key)
                        for key in (
                            "sequence_count",
                            "neff",
                            "neff_per_length",
                            "minimum_single_site_neff",
                            "minimum_site_effective_count",
                            "pairwise_minimum_neff_per_length",
                            "pseudocount_mode",
                            "pseudocount_value",
                            "cache_status",
                        )
                    }
            elif channel == "structure":
                field_map = {
                    "solvent_exposure": (
                        "mutation",
                        "status",
                        "sasa_angstrom2",
                        "relative_sasa",
                        "maximum_asa_reference",
                    ),
                    "contact_geometry": (
                        "mutation",
                        "status",
                        "contact_count",
                        "closest_contacts",
                    ),
                    "interface_contacts": (
                        "mutation",
                        "status",
                        "interface_contact_count",
                        "interface_contacts",
                    ),
                    "backbone_geometry": (
                        "mutation",
                        "status",
                        "phi_degrees",
                        "psi_degrees",
                        "secondary_structure",
                        "secondary_structure_method",
                        "missing_backbone_atoms",
                    ),
                    "interaction_flags": (
                        "mutation",
                        "status",
                        "hydrogen_bond_candidate_count",
                        "salt_bridge_candidate_count",
                        "disulfide_candidate_count",
                        "clash_candidate_count",
                        "mutant_side_chain_not_modelled",
                    ),
                }
                projected[focus] = {
                    key: {
                        name: value[name]
                        for name in field_map[focus]
                        if name in value
                    }
                    for key, value in sites.items()
                    if isinstance(value, Mapping)
                }
        retained = {
            key: evidence[key]
            for key in (
                "evidence_id",
                "round_id",
                "channel",
                "statement",
                "score",
                "source_id",
                "confidence",
                "evidence_type",
                "quality_status",
                "applicability",
                "uncertainty",
                "calibrated_score",
                "calibrated",
                "contributes_to_selection",
                "warnings",
                "provenance",
            )
            if key in evidence
        }
        retained["raw_features"] = projected
        retained["projection"] = list(projection)
        retained["positions"] = list(positions)
        return retained

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
