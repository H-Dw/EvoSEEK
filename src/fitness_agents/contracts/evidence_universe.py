"""One deterministic source of truth for evidence IDs visible to an agent role."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RoleVisibleEvidenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    evidence_id: str = Field(min_length=1, max_length=240)
    origins: tuple[str, ...]


class RoleVisibleEvidenceUniverse(BaseModel):
    """Exact evidence-ID universe presented to one role for one request.

    IDs are admitted by data flow, never by an ID prefix such as ``ev:local_rag``.
    The same object is used for prompt disclosure, deterministic validation and
    repair hints so those paths cannot silently disagree.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    role: str = Field(min_length=1, max_length=120)
    entries: tuple[RoleVisibleEvidenceEntry, ...] = ()

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(item.evidence_id for item in self.entries)

    def require_known(self, evidence_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted(set(evidence_ids).difference(self.ids)))

    def prompt_payload(self) -> dict[str, Any]:
        """Expose the exact ID authority without repeating per-ID origin labels."""

        return {
            "role": self.role,
            "allowed_evidence_ids": sorted(self.ids),
        }

    @classmethod
    def from_role_sources(
        cls,
        *,
        role: str,
        evidence: Iterable[Any] = (),
        interaction: Any | None = None,
        approved_channel_analyses: Iterable[Any] = (),
    ) -> RoleVisibleEvidenceUniverse:
        origins: dict[str, set[str]] = {}

        def add(raw_id: Any, origin: str) -> None:
            if raw_id is None or not str(raw_id):
                return
            origins.setdefault(str(raw_id), set()).add(origin)

        for item in evidence:
            add(_value(item, "evidence_id"), "evidence")

        for pack in _items(_value(interaction, "packs")):
            operator = str(_value(pack, "operator") or "unknown")
            for item in _items(_value(pack, "evidence")):
                add(_value(item, "evidence_id"), f"kg_pack:{operator}")

        for approved in approved_channel_analyses:
            channel = str(_value(approved, "channel") or "unknown")
            analysis = _value(approved, "hypothesis") or _value(approved, "analysis")
            for raw_id in _analysis_evidence_ids(analysis):
                add(raw_id, f"approved_channel_analysis:{channel}")

        return cls(
            role=role,
            entries=tuple(
                RoleVisibleEvidenceEntry(
                    evidence_id=evidence_id,
                    origins=tuple(sorted(item_origins)),
                )
                for evidence_id, item_origins in sorted(origins.items())
            ),
        )


def _value(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return ()


def _analysis_evidence_ids(analysis: Any) -> frozenset[str]:
    if analysis is None:
        return frozenset()
    ids = {str(item) for item in _items(_value(analysis, "evidence_ids")) if item}
    for finding in _items(_value(analysis, "findings")):
        ids.update(
            str(item) for item in _items(_value(finding, "evidence_ids")) if item
        )
    for hypothesis in _items(_value(analysis, "candidate_hypotheses")):
        ids.update(
            str(item) for item in _items(_value(hypothesis, "evidence_ids")) if item
        )
    return frozenset(ids)


__all__ = ["RoleVisibleEvidenceEntry", "RoleVisibleEvidenceUniverse"]
