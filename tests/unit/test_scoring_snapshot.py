from __future__ import annotations

import pytest

from fitness_agents.contracts.schemas import DesignScore, Prediction, Variant
from fitness_agents.contracts.scoring_snapshot import RoundScoringSnapshot, StateCoverageError


def _variant(variant_id: str) -> Variant:
    return Variant(variant_id, "VDGV", "VDGV", "WT", 0, "candidate")


def _prediction(variant_id: str) -> Prediction:
    return Prediction(variant_id, 0.2, 0.1, (0.0, 0.4), 0.0, {}, "test")


def _design(variant_id: str) -> DesignScore:
    return DesignScore(variant_id, 0.2, 0.1, 0.0, 0.0, 0.0, 0.2, "agent_uq", "reason")


def test_snapshot_reports_missing_map_and_version_instead_of_key_error() -> None:
    variant = _variant("v1")
    snapshot = RoundScoringSnapshot(
        hypothesis_id="hyp:2",
        version=3,
        eligible=(variant,),
        design_score_by_id={"v1": _design("v1")},
        prediction_by_id={"v1": _prediction("v1")},
        model_ranks={},
        all_scores={"v1": 0.2},
        acquisition_ranks={"v1": 1},
        eligible_ranks={"v1": 1},
    )
    with pytest.raises(StateCoverageError) as captured:
        snapshot.assert_selection_coverage(["v1"])
    assert captured.value.snapshot_version == "hyp:2:v3"
    assert captured.value.missing_by_field == {"model_ranks": ("v1",)}


def test_snapshot_rejects_selection_outside_current_eligible_universe() -> None:
    snapshot = RoundScoringSnapshot(
        hypothesis_id="hyp:2",
        version=2,
        eligible=(_variant("v1"),),
        design_score_by_id={"v1": _design("v1"), "v2": _design("v2")},
        prediction_by_id={"v1": _prediction("v1"), "v2": _prediction("v2")},
        model_ranks={"v1": 1, "v2": 2},
        all_scores={"v1": 0.2, "v2": 0.1},
        acquisition_ranks={"v1": 1, "v2": 2},
        eligible_ranks={"v1": 1, "v2": 2},
    )
    with pytest.raises(StateCoverageError) as captured:
        snapshot.assert_selection_coverage(["v2"])
    assert captured.value.missing_by_field == {"eligible": ("v2",)}


def test_snapshot_allows_pool_without_dry_predictions_until_selection() -> None:
    variants = (_variant("v1"), _variant("v2"))
    snapshot = RoundScoringSnapshot(
        hypothesis_id="hyp:2",
        version=4,
        eligible=variants,
        design_score_by_id={item.variant_id: _design(item.variant_id) for item in variants},
        prediction_by_id={"v1": _prediction("v1")},
        model_ranks={"v1": 1},
        all_scores={"v1": 0.2, "v2": 0.1},
        acquisition_ranks={"v1": 1, "v2": 2},
        eligible_ranks={"v1": 1, "v2": 2},
    )

    snapshot.assert_eligible_coverage()
    snapshot.assert_selection_coverage(["v1"])
    with pytest.raises(StateCoverageError) as captured:
        snapshot.assert_selection_coverage(["v2"])
    assert captured.value.missing_by_field == {
        "model_ranks": ("v2",),
        "prediction_by_id": ("v2",),
    }
