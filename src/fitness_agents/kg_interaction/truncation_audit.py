"""Keyword-item audit for structured-KG to bounded-LLM row truncation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import InteractionResult


@dataclass(frozen=True)
class KeywordTruncationEntry:
    item: str
    entity_match_count: int
    relation_match_count: int
    total_match_count: int
    returned_match_count: int
    max_rows: int
    truncated: bool
    status: str
    sample_matches: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class KGTruncationAuditReport:
    round_id: int
    max_rows: int
    entries: tuple[KeywordTruncationEntry, ...]

    @property
    def any_truncated(self) -> bool:
        return any(item.truncated for item in self.entries)

    @property
    def missing_items(self) -> tuple[str, ...]:
        return tuple(item.item for item in self.entries if item.status == "not_found")

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "max_rows": self.max_rows,
            "any_truncated": self.any_truncated,
            "missing_items": self.missing_items,
            "entries": tuple(asdict(item) for item in self.entries),
        }


class KGKeywordTruncationAuditor:
    """Compare exact KG keyword counts with the bounded rows available to a tool call."""

    def __init__(self, structured_sink: Any) -> None:
        if not hasattr(structured_sink, "query_keyword"):
            raise TypeError("structured_sink must provide query_keyword")
        self.structured_sink = structured_sink

    def audit(
        self,
        items: Sequence[str],
        *,
        round_id: int,
        max_rows: int,
        sample_rows: int = 3,
    ) -> KGTruncationAuditReport:
        if max_rows < 1 or sample_rows < 1:
            raise ValueError("max_rows and sample_rows must be positive")
        normalized = tuple(dict.fromkeys(str(item).strip() for item in items))
        if not normalized or any(not item for item in normalized):
            raise ValueError("items must contain at least one non-empty keyword")
        entries = []
        for item in normalized:
            result = self.structured_sink.query_keyword(
                item=item,
                round_id=round_id,
                limit=max_rows,
            )
            entries.append(
                KeywordTruncationEntry(
                    item=item,
                    entity_match_count=int(result["entity_match_count"]),
                    relation_match_count=int(result["relation_match_count"]),
                    total_match_count=int(result["total_match_count"]),
                    returned_match_count=int(result["returned_match_count"]),
                    max_rows=max_rows,
                    truncated=bool(result["truncated"]),
                    status=str(result["status"]),
                    sample_matches=tuple(result.get("matches", ()))[:sample_rows],
                )
            )
        return KGTruncationAuditReport(round_id, max_rows, tuple(entries))


def interaction_item_presence(
    interaction: InteractionResult,
    items: Sequence[str],
    *,
    excluded_operators: frozenset[str] = frozenset({"query_kg_truncation_audit"}),
) -> tuple[dict[str, Any], ...]:
    """Report whether normal, pre-audit tool packs already expose each keyword item."""

    searchable_packs = tuple(
        pack for pack in interaction.packs if pack.operator not in excluded_operators
    )
    pack_text = {
        pack.operator: json.dumps(asdict(pack), sort_keys=True, default=str).casefold()
        for pack in searchable_packs
    }
    output = []
    for item in items:
        keyword = str(item).strip().casefold()
        operators = tuple(
            operator for operator, text in pack_text.items() if keyword and keyword in text
        )
        output.append(
            {
                "item": str(item),
                "present_in_non_audit_packs": bool(operators),
                "matching_operators": operators,
            }
        )
    return tuple(output)


def runtime_truncation_audit_payload(
    interaction: InteractionResult,
    items: Sequence[str],
) -> dict[str, Any] | None:
    """Extract the audit tool report and add normal-pack visibility diagnostics."""

    audit_pack = next(
        (pack for pack in interaction.packs if pack.operator == "query_kg_truncation_audit"),
        None,
    )
    if audit_pack is None:
        return None
    report = dict(audit_pack.metadata.get("audit_report", {}))
    report["query_id"] = audit_pack.query_id
    report["interaction_presence"] = interaction_item_presence(interaction, items)
    return report
