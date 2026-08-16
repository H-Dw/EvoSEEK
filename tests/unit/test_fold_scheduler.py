from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from fitness_agents.config import project_root


def _load_scheduler_module():
    path = project_root() / "scripts/run_fold_campaigns.py"
    spec = importlib.util.spec_from_file_location("fitness_agents_fold_scheduler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_folds_supports_all_and_rejects_invalid_values():
    scheduler = _load_scheduler_module()
    assert scheduler._parse_folds("all", 5) == [0, 1, 2, 3, 4]
    assert scheduler._parse_folds("4,1", 5) == [4, 1]
    with pytest.raises(ValueError, match="duplicates"):
        scheduler._parse_folds("1,1", 5)
    with pytest.raises(ValueError, match="outside"):
        scheduler._parse_folds("5", 5)


def test_fold_jobs_respect_parallel_limit_and_collect_all_results(tmp_path, monkeypatch):
    scheduler = _load_scheduler_module()
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_run(command, **_kwargs):
        nonlocal active, max_active
        fold = int(command[command.index("--fold-index") + 1])
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        summary = {
            "run_id": f"run-fold-{fold}",
            "mode": "knowledge_agent",
            "seed": 11,
            "round_metrics": [
                {
                    "best_seen_fitness": 1.0,
                    "batch_mean_fitness": 0.5,
                    "mean_selected_model_rank_fraction": 0.1,
                }
            ],
            "final_prediction_metrics": {"spearman": 0.7},
            "queries_used": 2,
            "data_source": {"fold_index": fold},
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(summary), "")

    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
    jobs = scheduler._build_jobs(
        config_path=Path("config.yaml"),
        folds=[0, 1, 2, 3, 4],
        seed=11,
        python_executable=sys.executable,
    )
    results = scheduler.run_fold_jobs(
        jobs,
        max_parallel=2,
        project_dir=tmp_path,
        output_dir=tmp_path / "logs",
    )
    assert max_active == 2
    assert [result.fold_index for result in results] == [0, 1, 2, 3, 4]
    assert all(result.status == "completed" for result in results)
    assert len(list((tmp_path / "logs").glob("*.stdout.log"))) == 5


def test_fold_job_failure_is_recorded_without_losing_other_folds(tmp_path, monkeypatch):
    scheduler = _load_scheduler_module()

    def fake_run(command, **_kwargs):
        fold = int(command[command.index("--fold-index") + 1])
        if fold == 1:
            return subprocess.CompletedProcess(command, 9, "", "simulated failure")
        summary = {
            "run_id": f"run-fold-{fold}",
            "mode": "knowledge_agent",
            "seed": 3,
            "round_metrics": [{"best_seen_fitness": 1.0}],
            "final_prediction_metrics": {},
            "queries_used": 1,
            "data_source": {"fold_index": fold},
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(summary), "")

    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
    jobs = scheduler._build_jobs(
        config_path=Path("config.yaml"),
        folds=[0, 1, 2],
        seed=3,
        python_executable=sys.executable,
    )
    results = scheduler.run_fold_jobs(
        jobs,
        max_parallel=3,
        project_dir=tmp_path,
        output_dir=tmp_path / "logs",
    )
    assert [result.status for result in results] == ["completed", "failed", "completed"]
    assert results[1].returncode == 9
    assert "simulated failure" in Path(results[1].stderr_log).read_text(encoding="utf-8")


def test_aborted_fold_is_failed_even_when_stdout_is_a_summary(tmp_path, monkeypatch):
    scheduler = _load_scheduler_module()

    def fake_run(command, **_kwargs):
        fold = int(command[command.index("--fold-index") + 1])
        summary = {
            "run_id": f"run-fold-{fold}",
            "mode": "knowledge_agent",
            "seed": 42,
            "round_metrics": [],
            "final_prediction_metrics": None,
            "queries_used": 0,
            "rounds_aborted": 1,
            "data_source": {"fold_index": fold},
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(summary), "")

    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
    jobs = scheduler._build_jobs(
        config_path=Path("config.yaml"),
        folds=[0],
        seed=42,
        python_executable=sys.executable,
    )
    results = scheduler.run_fold_jobs(
        jobs,
        max_parallel=1,
        project_dir=tmp_path,
        output_dir=tmp_path / "logs",
    )
    assert results[0].status == "failed"
    assert "aborted" in (results[0].error or "")


def test_aggregate_runs_skips_summaries_without_round_metrics(tmp_path):
    from fitness_agents.reporting import aggregate_runs

    paths = aggregate_runs(
        [
            {
                "run_id": "aborted",
                "mode": "knowledge_agent",
                "seed": 1,
                "queries_used": 0,
                "round_metrics": [],
                "final_prediction_metrics": None,
                "data_source": {"fold_index": 0},
            },
            {
                "run_id": "ok",
                "mode": "knowledge_agent",
                "seed": 1,
                "queries_used": 4,
                "round_metrics": [
                    {
                        "best_seen_fitness": 1.2,
                        "batch_mean_fitness": 0.8,
                        "mean_selected_model_rank_fraction": 0.2,
                    }
                ],
                "final_prediction_metrics": {"spearman": 0.4},
                "data_source": {"fold_index": 1},
            },
        ],
        tmp_path / "aggregate",
    )
    rows = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert [row["run_id"] for row in rows] == ["ok"]
