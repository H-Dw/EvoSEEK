from __future__ import annotations

import pytest

from fitness_agents.agents.short_ids import (
    AmbiguousLocalIdError,
    FieldIdPolicy,
    RequestScopedIdBridge,
    UnresolvedLocalIdError,
)


def _bridge() -> RequestScopedIdBridge:
    return RequestScopedIdBridge.build(
        scope_id="IDB000001",
        role="critic",
        schema_name="Example",
        namespace_values={
            "S": ("canonical:sample:alpha", "canonical:sample:beta"),
            "E": ("canonical:evidence:one", "canonical:evidence:two"),
            "D": ("canonical:decision:one",),
        },
        field_policies={
            "items[].sample_id": FieldIdPolicy("S", "unique_near"),
            "items[].evidence_ids[]": FieldIdPolicy("E", "normalize"),
            "decision_id": FieldIdPolicy("D", "exact"),
        },
    )


def test_bridge_projects_only_declared_id_paths_and_round_trips() -> None:
    bridge = _bridge()
    projected = bridge.encode_projection(
        {
            "items": [
                {
                    "sample_id": "canonical:sample:alpha",
                    "evidence_ids": ["canonical:evidence:one"],
                    "explanation": "canonical:sample:alpha remains ordinary prose here",
                }
            ],
            "decision_id": "canonical:decision:one",
        }
    )

    assert projected["items"][0]["sample_id"] == "S01"
    assert projected["items"][0]["evidence_ids"] == ["E01"]
    assert projected["items"][0]["explanation"].startswith("canonical:sample")
    assert projected["decision_id"] == "D01"
    assert bridge.decode_and_validate(projected) == {
        "items": [
            {
                "sample_id": "canonical:sample:alpha",
                "evidence_ids": ["canonical:evidence:one"],
                "explanation": "canonical:sample:alpha remains ordinary prose here",
            }
        ],
        "decision_id": "canonical:decision:one",
    }


def test_bridge_normalizes_formatting_but_keeps_high_risk_ids_exact() -> None:
    bridge = _bridge()
    decoded = bridge.decode_and_validate(
        {
            "items": [{"sample_id": "Ｓ１", "evidence_ids": ["`e1`"]}],
            "decision_id": "D01",
        }
    )

    assert decoded["items"][0]["sample_id"] == "canonical:sample:alpha"
    assert decoded["items"][0]["evidence_ids"] == ["canonical:evidence:one"]
    assert any(item.corrected for item in bridge.resolution_receipts)
    with pytest.raises(UnresolvedLocalIdError):
        bridge.decode_and_validate(
            {
                "items": [{"sample_id": "S01", "evidence_ids": ["E01"]}],
                "decision_id": "d01",
            }
        )


def test_bridge_never_guesses_between_multiple_near_candidates() -> None:
    bridge = RequestScopedIdBridge.build(
        scope_id="IDB000002",
        role="critic",
        schema_name="Example",
        namespace_values={"S": tuple(f"canonical:{index}" for index in range(1, 13))},
        field_policies={"sample_id": FieldIdPolicy("S", "unique_near")},
    )
    with pytest.raises(AmbiguousLocalIdError):
        bridge.decode_and_validate({"sample_id": "S0X"})


def test_bridge_rejects_unknown_canonical_or_local_ids() -> None:
    bridge = _bridge()
    with pytest.raises(UnresolvedLocalIdError):
        bridge.encode_projection(
            {
                "items": [{"sample_id": "canonical:outside", "evidence_ids": []}],
                "decision_id": "canonical:decision:one",
            }
        )
    with pytest.raises(UnresolvedLocalIdError):
        bridge.decode_and_validate(
            {
                "items": [{"sample_id": "S99", "evidence_ids": []}],
                "decision_id": "D01",
            }
        )
