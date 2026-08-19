"""Campaign-level substitution tables shared by conservation, structure, and physchem."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CANONICAL_RESIDUES = tuple("ACDEFGHIKLMNPQRSTVWY")
STATIC_FEATURE_CHANNELS = frozenset({"physchem", "conservation", "structure"})
STRUCTURE_CONTACT_LIST_KEYS = frozenset({"closest_contacts", "interface_contacts"})


def compact_static_evidence_id(
    channel: str,
    variant_id: str,
    parameter_set_id: str,
    resource_sha256: str,
) -> str:
    """Stable Evidence ID for static channels: no raw_features blob and no round_id."""

    payload = json.dumps(
        {
            "channel": channel,
            "variant_id": variant_id,
            "parameter_set_id": parameter_set_id,
            "resource_sha256": resource_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"ev:{channel}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def compact_structure_site(feature: Mapping[str, Any]) -> dict[str, Any]:
    """Copy WT environment scalars; omit bulky contact lists from variant evidence and KG."""

    return {
        key: value
        for key, value in feature.items()
        if key not in STRUCTURE_CONTACT_LIST_KEYS
    }


@dataclass(frozen=True)
class SubstitutionFeatureStore:
    """Replacement-level tables keyed by (position, wt, mutant), not combinatorial variants."""

    tables: dict[str, dict[str, Any]]

    @classmethod
    def from_providers(cls, providers: Mapping[str, object]) -> SubstitutionFeatureStore:
        tables: dict[str, dict[str, Any]] = {}
        for channel, provider in providers.items():
            getter = getattr(provider, "site_table", None)
            if not callable(getter):
                continue
            table = getter()
            if table:
                tables[str(channel)] = table
        return cls(tables)
