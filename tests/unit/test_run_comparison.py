from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fitness_agents.reporting.aggregate import (
    aggregate_runs,
    infer_condition,
    write_paired_contrast,
)


def _summary(
    run_id: str,
    *,
    mode: str,
    fold: int,
    best: float,
    condition: str | None = None,
    batch: float | None = None,
    allow_remote: bool | None = None,
):
    return {
        "run_id": run_id,
        "mode": mode,
        "condition": condition or mode,
        "seed": 42,
        "queries_used": 96,
        "allow_remote_context": allow_remote,
        "round_metrics": [
            {
                "best_seen_fitness": best,
                "batch_mean_fitness": best if batch is None else batch,
                "mean_selected_model_rank_fraction": 0.2,
            }
        ],
        "final_prediction_metrics": {},
        "data_source": {
            "strategy": "al96_closed_loop",
            "fold_index": fold,
            "assignment_sha256": f"fold-{fold}",
        },
    }


def test_infer_condition_prefers_explicit_then_rag_run_id():
    assert infer_condition({"condition": "knowledge_agent_rag", "mode": "knowledge_agent"}) == (
        "knowledge_agent_rag"
    )
    assert infer_condition(
        {
            "mode": "knowledge_agent",
            "run_id": "knowledge_agent-s42-f00-al96-knowledge_agent_rag",
        }
    ) == "knowledge_agent_rag"
    assert infer_condition({"mode": "knowledge_agent"}, job_mode="knowledge_agent_rag") == (
        "knowledge_agent_rag"
    )


def test_aggregate_runs_pairs_rag_against_same_fold_knowledge_agent(tmp_path: Path):
    paths = aggregate_runs(
        [
            _summary("kg-f0", mode="knowledge_agent", fold=0, best=5.0, allow_remote=False),
            _summary(
                "rag-f0",
                mode="knowledge_agent",
                condition="knowledge_agent_rag",
                fold=0,
                best=5.4,
                allow_remote=True,
            ),
            _summary("kg-f1", mode="knowledge_agent", fold=1, best=4.8, allow_remote=False),
        ],
        tmp_path,
    )
    rows = {item["run_id"]: item for item in json.loads(paths["json"].read_text())}
    assert rows["rag-f0"]["delta_best_seen_vs_knowledge_agent"] == pytest.approx(0.4)
    assert rows["kg-f0"]["delta_best_seen_vs_knowledge_agent"] == pytest.approx(0.0)
    assert rows["kg-f1"]["same_fold_knowledge_agent_available"] is True
    contrast = json.loads((tmp_path / "rag_contrast.json").read_text())
    assert len(contrast) == 1
    assert contrast[0]["delta_best_seen_rag_minus_no_rag"] == pytest.approx(0.4)
    assert (tmp_path / "rag_contrast.md").is_file()


def test_write_paired_contrast_is_empty_without_both_conditions(tmp_path: Path):
    paths = write_paired_contrast(
        [
            {
                "condition": "knowledge_agent",
                "seed": 1,
                "queries_used": 4,
                "split_strategy": "al96_closed_loop",
                "fold_index": 0,
                "assignment_sha256": "x",
                "run_id": "only",
                "best_seen_fitness": 1.0,
                "last_batch_mean_fitness": 0.5,
            }
        ],
        tmp_path,
    )
    assert json.loads(paths["rag_json"].read_text()) == []


def test_comparison_rag_dry_run_keeps_rag_flags_distinct():
    from fitness_agents.config import project_root

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_agent_baselines.py",
            "--preset",
            "al96",
            "--comparison",
            "rag",
            "--folds",
            "0",
            "--seeds",
            "42",
            "--max-parallel",
            "4",
            "--dry-run",
        ],
        cwd=project_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    jobs = payload["jobs"]
    assert [job["condition"] for job in jobs] == ["knowledge_agent", "knowledge_agent_rag"]
    by_condition = {job["condition"]: job for job in jobs}
    assert by_condition["knowledge_agent"]["rag_in_remote_context"] is False
    assert by_condition["knowledge_agent_rag"]["rag_in_remote_context"] is True
    assert by_condition["knowledge_agent_rag"]["allow_remote_context"] is True
    assert by_condition["knowledge_agent_rag"]["max_tool_calls"] >= (
        by_condition["knowledge_agent"]["max_tool_calls"]
    )
    assert payload["max_parallel"] == 4
    rag_command = by_condition["knowledge_agent_rag"]["command"]
    assert "-m" in rag_command and "fitness_agents.cli" in rag_command
    assert "--local-knowledge-index" in rag_command
    assert "--local-knowledge-index" not in by_condition["knowledge_agent"]["command"]


def test_qwen_rag_dry_run_uses_deepseek_llm_and_shared_api_corpus():
    from fitness_agents.config import project_root

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_agent_baselines.py",
            "--preset",
            "al96",
            "--comparison",
            "llm_vs_qwen_rag",
            "--folds",
            "0,1,2",
            "--seeds",
            "42",
            "--max-parallel",
            "1",
            "--dry-run",
        ],
        cwd=project_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    jobs = payload["jobs"]
    assert payload["modes"] == ["llm_agent", "knowledge_agent_qwen_rag"]
    assert len(jobs) == 6
    qwen_jobs = [job for job in jobs if job["condition"] == "knowledge_agent_qwen_rag"]
    llm_jobs = [job for job in jobs if job["condition"] == "llm_agent"]
    assert len(qwen_jobs) == 3
    assert len(llm_jobs) == 3
    command = qwen_jobs[0]["command"]
    assert qwen_jobs[0]["provider"] == "deepseek"
    assert qwen_jobs[0]["rag_in_remote_context"] is True
    assert "--local-knowledge-overlay" in command
    assert "--local-knowledge-index" not in command
    assert "--condition" in command
    assert llm_jobs[0]["knowledge_enabled"] is False
    assert "--local-knowledge-overlay" not in llm_jobs[0]["command"]


def _load_baseline_module():
    from fitness_agents.config import project_root

    path = project_root() / "scripts/run_agent_baselines.py"
    spec = importlib.util.spec_from_file_location("fitness_agents_run_agent_baselines", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_summary(command):
    fold = int(command[command.index("--fold-index") + 1])
    condition = command[command.index("--condition") + 1]
    return {
        "run_id": f"{condition}-f{fold:02d}",
        "mode": "knowledge_agent",
        "condition": condition,
        "seed": 42,
        "round_metrics": [
            {
                "best_seen_fitness": 1.0,
                "batch_mean_fitness": 0.5,
                "mean_selected_model_rank_fraction": 0.1,
            }
        ],
        "final_prediction_metrics": {},
        "queries_used": 2,
        "rounds_aborted": 0,
        "data_source": {"fold_index": fold},
    }


def test_baseline_jobs_respect_max_parallel_and_keep_going_after_failure(tmp_path, monkeypatch):
    import threading
    import time

    module = _load_baseline_module()
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_run(command, **_kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        condition = command[command.index("--condition") + 1]
        fold = int(command[command.index("--fold-index") + 1])
        if condition == "knowledge_agent" and fold == 1:
            return subprocess.CompletedProcess(command, 9, "", "simulated failure")
        return subprocess.CompletedProcess(command, 0, json.dumps(_fake_summary(command)), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    jobs = [
        module.BaselineJob(
            index=index,
            public={
                "condition": condition,
                "seed": 42,
                "fold_index": fold,
            },
            command=(
                sys.executable,
                "-m",
                "fitness_agents.cli",
                "config.yaml",
                "--fold-index",
                str(fold),
                "--condition",
                condition,
            ),
        )
        for index, (condition, fold) in enumerate(
            [
                ("knowledge_agent", 0),
                ("knowledge_agent_rag", 0),
                ("knowledge_agent", 1),
                ("knowledge_agent_rag", 1),
            ]
        )
    ]
    results = module.run_baseline_jobs(
        jobs,
        max_parallel=4,
        project_dir=tmp_path,
        output_dir=tmp_path / "logs",
    )
    assert max_active == 4
    assert [item.status for item in results] == [
        "completed",
        "completed",
        "failed",
        "completed",
    ]
    assert results[2].error is not None

