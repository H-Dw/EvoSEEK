from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from fitness_agents.config import load_experiment_config
from fitness_agents.contracts.schemas import FitnessObservation, Prediction, Variant
from fitness_agents.loop import CampaignRunner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/gb1_rethink_hypothesis_smoke.yaml"
ORACLE_PATH = PROJECT_ROOT / "data/demo/gb1_demo_oracle.csv"


def _benchmark_truth() -> dict[str, float]:
    with ORACLE_PATH.open(encoding="utf-8", newline="") as handle:
        return {row["variant_id"]: float(row["fitness"]) for row in csv.DictReader(handle)}


class _BenchmarkTruthPredictor:
    """Test-only final evaluator that replays GB1 benchmark measurements."""

    model_version = "gb1-benchmark-truth:v1"

    def __init__(self, truth: dict[str, float]) -> None:
        self._truth = truth
        self.fit_calls = 0

    def fit(
        self,
        variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
        validation_variants: Sequence[Variant] | None = None,
        validation_observations: Sequence[FitnessObservation] | None = None,
    ) -> _BenchmarkTruthPredictor:
        del variants, observations, validation_variants, validation_observations
        self.fit_calls += 1
        return self

    def predict(self, variants: Sequence[Variant]) -> list[Prediction]:
        predictions: list[Prediction] = []
        for variant in variants:
            value = self._truth[variant.variant_id]
            predictions.append(
                Prediction(
                    variant_id=variant.variant_id,
                    fitness_mean=value,
                    fitness_std=0.0,
                    interval_90=(value, value),
                    ood_score=0.0,
                    component_scores={"benchmark_truth": value},
                    model_version=self.model_version,
                    is_measured=True,
                )
            )
        return predictions


@pytest.mark.integration
def test_gb1_hypothesis_rethink_completes_three_wet_only_rounds(tmp_path: Path) -> None:
    truth = _benchmark_truth()
    created_predictors: list[_BenchmarkTruthPredictor] = []
    factory_calls: list[dict[str, Any]] = []

    def benchmark_truth_factory(_model_config: Any, *, seed: int) -> _BenchmarkTruthPredictor:
        predictor = _BenchmarkTruthPredictor(truth)
        created_predictors.append(predictor)
        factory_calls.append({"seed": seed})
        return predictor

    config = load_experiment_config(CONFIG_PATH)
    assert config.rounds == 3
    assert config.generation.use_fitness_predictors is False
    assert config.validation.enabled is False
    assert config.validation.rethink_enabled is True
    assert config.validation.rethink_mode == "hypothesis"
    config = replace(config, output_root=tmp_path / "runs")

    summary = CampaignRunner(
        config,
        predictor_factory=benchmark_truth_factory,
    ).run()
    run_dir = Path(summary["run_dir"])

    assert summary["run_status"] == "completed"
    assert summary["rounds_aborted"] == 0
    assert summary["planned_batch_sizes"] == [4, 4, 4]
    assert summary["actual_batch_sizes"] == [4, 4, 4]
    assert summary["queries_used"] == 12
    assert summary["fitness_predictors_used_for_generation"] is False
    assert summary["rethink_reflections"] == 0
    assert summary["hypothesis_reflections"] == 3
    assert summary["required_node_failures"] == []
    assert summary["fallback_nodes"] == []

    # Generation and round validation create no predictors. The sole factory
    # call is the benchmark-truth adapter required by final evaluation.
    assert factory_calls == [{"seed": config.seed + config.rounds + 1}]
    assert len(created_predictors) == 1
    assert created_predictors[0].fit_calls == 1
    assert summary["final_prediction_metrics"]["mse"] == pytest.approx(0.0)
    assert summary["final_prediction_metrics"]["rmse"] == pytest.approx(0.0)

    for round_id in range(1, 4):
        round_dir = run_dir / f"round_{round_id:02d}"
        validation_records = json.loads(
            (round_dir / "validation_matrix.json").read_text(encoding="utf-8")
        )
        assert len(validation_records) == config.budget_per_round
        assert {item["validation_type"] for item in validation_records} == {"wet"}
        assert all(item["model_version"] is None for item in validation_records)
        assert all(item["reflection_id"] is None for item in validation_records)
        assert all(item["reflection_summary"] == "" for item in validation_records)
        assert all(
            item["value"] == pytest.approx(truth[item["variant_id"]]) for item in validation_records
        )

        reflection = json.loads(
            (round_dir / "hypothesis_reflection.json").read_text(encoding="utf-8")
        )
        assert reflection.get("status") != "NOT_APPLICABLE"
        assert reflection["round_id"] == round_id
        assert reflection["hypothesis_id"]
        assert reflection["assessment_id"]
        assert reflection["advisory_only"] is True
        assert reflection["selection_eligible"] is False

    trace = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(item.get("event_type") == "rethink_completed" for item in trace) == 3

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert len(state["hypothesis_reflections"]) == 3
    assert {item["round_id"] for item in state["hypothesis_reflections"]} == {1, 2, 3}
