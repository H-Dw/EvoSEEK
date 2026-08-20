from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from fitness_agents.config import load_experiment_config, project_root


def _load_runner():
    path = project_root() / "scripts/run_hierarchical_scientist.py"
    spec = importlib.util.spec_from_file_location("fitness_agents_hierarchical_scientist_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _passing_completion_manifest(*, expected_rounds: int = 3) -> dict:
    return {
        "artifact_finalized": True,
        "run_status": "completed",
        "experiment_status": "completed",
        "evaluation_status": "eligible",
        "pass_eligible": True,
        "expected_rounds": expected_rounds,
        "completed_rounds": expected_rounds,
        "aborted_rounds": 0,
        "planned_batch_sizes": [16] * expected_rounds,
        "actual_batch_sizes": [16] * expected_rounds,
        "required_node_failures": [],
        "fallback_nodes": [],
    }


def _passing_pipeline() -> dict:
    return {
        "status": "SUCCEEDED",
        "branches": [
            {"channel": "physchem", "status": "SUCCEEDED", "attempts": 1},
            {"channel": "conservation", "status": "SUCCEEDED", "attempts": 1},
            {"channel": "structure", "status": "SUCCEEDED", "attempts": 1},
        ],
        "conflicts": [],
        "main_review": {
            "review_scope": "main",
            "decision_id": "mainreview:test",
            "verdict": "APPROVE",
            "issues": [],
            "required_changes": [],
            "cited_evidence_ids": [],
            "summary": "Approved isolated channel synthesis.",
        },
        "main_attempts": 1,
    }


def _write_campaign(
    run_dir: Path,
    *,
    fold_index: int,
    condition: str = "hierarchical",
    rounds: int = 3,
    budget: int = 16,
    candidate_limit: int = 32,
    write_pipeline: bool = True,
    hierarchy_enabled: bool | None = None,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "hierarchical": {"channels": True, "rag": False, "al": False, "hierarchy": True},
        "single": {"channels": True, "rag": False, "al": False, "hierarchy": False},
        "kg_base": {"channels": False, "rag": False, "al": False, "hierarchy": False},
        "kg_base_rag": {"channels": False, "rag": True, "al": False, "hierarchy": False},
        "kg_base_al": {"channels": False, "rag": False, "al": True, "hierarchy": False},
        "kg_3features_rag": {"channels": True, "rag": True, "al": False, "hierarchy": True},
    }[condition]
    enabled = spec["hierarchy"] if hierarchy_enabled is None else hierarchy_enabled
    operators = ["hypothesis_context", "query_assay_association", "query_evidence_provenance"]
    if spec["channels"]:
        operators.append("query_feature_bundle")
    operators.append("query_kg_truncation_audit")
    if spec["rag"]:
        operators.extend(["query_local_knowledge", "query_structured_claims"])
    operators.extend(["explain_variant", "compare_variants"])
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "llm_provider": "deepseek",
                "rounds": rounds,
                "budget_per_round": budget,
                "candidate_limit": candidate_limit,
                "hierarchical_hypothesis": {
                    "enabled": enabled,
                    "required_channels": ["physchem", "conservation", "structure"],
                    "child_max_tokens": 20000,
                    "child_critic_max_tokens": 20000,
                    "main_critic_max_tokens": 20000,
                },
                "llm": {"max_tokens": 20000},
                "critic": {"max_tokens": 20000},
                "model": "kermut",
                "validation": {"enabled": True},
                "knowledge_channels": {
                    "physchem": spec["channels"],
                    "conservation": spec["channels"],
                    "structure": spec["channels"],
                    "kg": True,
                },
                "kg_interaction": {
                    "feature_tool_strategy": (
                        "independent_and_joint" if spec["channels"] else "context_only"
                    ),
                    "enabled_operators": operators,
                },
                "knowledge_runtime": {
                    "local_knowledge": {
                        "enabled": spec["rag"],
                        "scientist_context_allowed": spec["rag"],
                    }
                },
                "generation": {
                    "selection_driver": "active_learning" if spec["al"] else "agent_uq",
                    "use_fitness_predictors": False,
                    "predictor_models": [],
                    "quota_allocation": {"enabled": not spec["al"]},
                },
                "active_learning": {
                    "enabled": spec["al"],
                    "posterior_models": ["kermut"],
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "completion_manifest.json").write_text(
        json.dumps(_passing_completion_manifest(expected_rounds=rounds)),
        encoding="utf-8",
    )
    if write_pipeline:
        for round_id in range(1, rounds + 1):
            round_dir = run_dir / f"round_{round_id:02d}"
            round_dir.mkdir()
            selected_ids = [f"v{index}" for index in range(budget)]
            (round_dir / "prediction_scope_receipt.json").write_text(
                json.dumps(
                    {
                        "planned_candidate_count": candidate_limit,
                        "round_candidate_count": candidate_limit,
                        "acquisition_prediction_count": (
                            candidate_limit if spec["al"] else 0
                        ),
                        "acquisition_predictions_within_round_candidate_set": True,
                        "approved_batch_size": budget,
                        "dry_validation_scope": "draft_selected_candidates_only",
                        "dry_validation_calls": [
                            {
                                "candidate_ids": selected_ids,
                                "candidate_count": budget,
                            }
                        ],
                        "all_dry_validation_targets_were_draft_selected": True,
                        "acquisition_prediction_scope": (
                            "candidate_pool" if spec["al"] else "none"
                        ),
                        "oracle_measurement_scope": "approved_batch_only",
                    }
                ),
                encoding="utf-8",
            )
            (round_dir / "hypothesis_pipeline.json").write_text(
                json.dumps(_passing_pipeline()),
                encoding="utf-8",
            )
    summary = {
        "run_id": f"run-{condition}-f{fold_index:02d}",
        "run_dir": str(run_dir),
        "mode": "knowledge_agent",
        "seed": 11,
        "condition": condition,
        "round_metrics": [
            {"round_id": index, "best_seen_fitness": 1.0} for index in range(1, rounds + 1)
        ],
        "final_prediction_metrics": {"spearman": 0.4},
        "queries_used": budget * rounds,
        "rounds_aborted": 0,
        "placeholder_predictor": False,
        "fitness_predictors_used_for_generation": spec["al"],
        "data_source": {"fold_index": fold_index, "assignment_sha256": f"assign-fold-{fold_index}"},
    }
    return summary


def test_hierarchical_scientist_config_matches_formal_al96_protocol() -> None:
    config = load_experiment_config("configs/experiments/hierarchical_scientist.deepseek.yaml")
    hierarchy = config.hierarchical_hypothesis
    assert config.rounds == 3
    assert config.budget_per_round == 16
    assert config.candidate_limit == 32
    assert config.task.split_root is not None
    assert config.llm.model == "deepseek-v4-flash"
    assert config.llm.api_key == "env:DEEPSEEK_API_KEY"
    assert config.critic.review_controls is True
    assert config.critic.review_diversity is True
    assert config.knowledge.local_knowledge.enabled is False
    assert config.model.name == "kermut"
    assert config.generation.use_fitness_predictors is False
    assert config.generation.predictor_models == ()
    assert config.generation.predictor_weight == 0.0
    assert [item.name for item in config.active_learning.posterior.predictor_models] == [
        "kermut"
    ]
    assert config.validation.enabled is True
    assert config.llm.max_tokens == 32768
    assert config.critic.max_tokens == 32768
    assert hierarchy.child_max_tokens == 32768
    assert hierarchy.child_critic_max_tokens == 32768
    assert hierarchy.main_critic_max_tokens == 32768
    assert hierarchy.enabled is True
    assert hierarchy.required_channels == ("physchem", "conservation", "structure")
    assert hierarchy.max_parallel_branches == 3
    assert hierarchy.child_sample_batch_size == 8
    assert hierarchy.child_max_parallel_batches == 2
    assert hierarchy.max_child_revision_attempts == 1
    assert hierarchy.max_main_revision_attempts == 2
    assert config.kg_interaction.max_tool_calls == 18
    assert config.scientist_prompt_evidence_limit == 32
    assert hierarchy.main_max_input_chars == 160000
    assert hierarchy.child_max_input_chars == 120000
    assert hierarchy.critic_max_input_chars == 120000
    assert hierarchy.subcritic_mode == "remote"
    assert config.kg_interaction.feature_channels == ("physchem", "conservation", "structure")
    assert config.generation.quota_allocation.quotas() == {
        "hypothesis_target": 8,
        "evidence_prior": 3,
        "coverage_exploration": 3,
        "matched_control": 2,
    }


def test_canary_placeholder_predictor_is_deterministic_and_explicitly_labeled() -> None:
    runner = _load_runner()
    variants = [
        type("Variant", (), {"variant_id": "v1"})(),
        type("Variant", (), {"variant_id": "v2"})(),
    ]
    first = runner._canary_placeholder_predictor_factory(None, seed=11)
    second = runner._canary_placeholder_predictor_factory(None, seed=11)

    first_predictions = first.predict(variants)
    second_predictions = second.predict(variants)

    assert first_predictions == second_predictions
    assert all(
        item.model_version == "placeholder-canary-sha256-seed11"
        for item in first_predictions
    )


def test_parse_folds_and_conditions_reject_invalid_values() -> None:
    runner = _load_runner()
    assert runner._parse_folds("all", 5) == [0, 1, 2, 3, 4]
    assert runner._parse_folds("0,1,2", 5) == [0, 1, 2]
    with pytest.raises(ValueError, match="duplicates"):
        runner._parse_folds("1,1", 5)
    with pytest.raises(ValueError, match="outside"):
        runner._parse_folds("5", 5)
    assert runner._parse_conditions("hierarchical") == ["hierarchical"]
    assert runner._parse_conditions("hierarchical,single") == ["hierarchical", "single"]
    assert runner._parse_conditions("kg_base,kg_base_rag,kg_base_al,kg_3features_rag") == [
        "kg_base",
        "kg_base_rag",
        "kg_base_al",
        "kg_3features_rag",
    ]
    with pytest.raises(ValueError, match="Unknown"):
        runner._parse_conditions("rag_kg_all")


def test_dry_run_schedule_has_three_same_task_fold_jobs(tmp_path, monkeypatch, capsys) -> None:
    runner = _load_runner()
    split_root = tmp_path / "split"
    split_root.mkdir()
    (split_root / "manifest.public.json").write_text(
        json.dumps(
            {
                "n_folds": 5,
                "strategy": "al96_closed_loop",
                "protocol_version": "GB1-AL96-5CV-v1",
            }
        ),
        encoding="utf-8",
    )
    base = load_experiment_config("configs/experiments/hierarchical_scientist.deepseek.yaml")
    real_load = runner.load_experiment_config

    def fake_load(path, overrides=None):
        if Path(path).name == "hierarchical_scientist.deepseek.yaml":
            return replace(base, task=replace(base.task, split_root=split_root))
        return real_load(path, overrides=overrides)

    monkeypatch.setattr(runner, "load_experiment_config", fake_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hierarchical_scientist.py",
            "--dry-run",
            "--folds",
            "0,1,2",
            "--conditions",
            "hierarchical",
            "--max-parallel",
            "3",
        ],
    )
    runner.main()
    schedule = json.loads(capsys.readouterr().out)
    assert schedule["schema_version"] == "hierarchical-scientist-schedule:v1"
    assert schedule["conditions"] == ["hierarchical"]
    assert schedule["folds"] == [0, 1, 2]
    assert schedule["max_parallel"] == 3
    assert schedule["expected_candidate_pool"] == 32
    assert "routes" not in schedule
    jobs = schedule["jobs"]
    assert len(jobs) == 3
    assert [job["condition"] for job in jobs] == ["hierarchical", "hierarchical", "hierarchical"]
    assert [job["fold_index"] for job in jobs] == [0, 1, 2]
    assert len({tuple(job["command"]) for job in jobs}) == 3
    assert all(
        job["command"][job["command"].index("--worker-candidate-limit") + 1]
        == "32"
        for job in jobs
    )


def test_same_task_folds_respect_parallel_limit(tmp_path, monkeypatch) -> None:
    runner = _load_runner()
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_run(command, **_kwargs):
        nonlocal active, max_active
        fold = int(command[command.index("--worker-fold") + 1])
        condition = command[command.index("--worker-condition") + 1]
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        summary = _write_campaign(tmp_path / f"run-f{fold:02d}", fold_index=fold, condition=condition)
        return subprocess.CompletedProcess(command, 0, json.dumps(summary), "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    jobs = runner._build_jobs(
        script_path=Path("scripts/run_hierarchical_scientist.py"),
        config_path=Path("config.yaml"),
        conditions=["hierarchical"],
        folds=[0, 1, 2],
        seed=11,
        expected_rounds=3,
        expected_budget=16,
        expected_candidate_limit=32,
        output_root=tmp_path / "runs",
        python_executable=sys.executable,
    )
    results = runner.run_fold_jobs(
        jobs,
        max_parallel=2,
        project_dir=tmp_path,
        output_dir=tmp_path / "logs",
    )
    assert max_active == 2
    assert [result.fold_index for result in results] == [0, 1, 2]
    assert {result.condition for result in results} == {"hierarchical"}
    assert all(result.status == "passed" for result in results)
    assert len(list((tmp_path / "logs").glob("*.stdout.log"))) == 3


def test_parallel_jobs_pin_distinct_cuda_devices(tmp_path, monkeypatch) -> None:
    runner = _load_runner()
    lock = threading.Lock()
    borrowed: set[str] = set()
    max_borrowed = 0
    seen: list[str] = []

    def fake_run(command, **kwargs):
        nonlocal max_borrowed
        env = kwargs["env"]
        device = env["CUDA_VISIBLE_DEVICES"]
        fold = int(command[command.index("--worker-fold") + 1])
        condition = command[command.index("--worker-condition") + 1]
        with lock:
            assert device not in borrowed
            borrowed.add(device)
            seen.append(device)
            max_borrowed = max(max_borrowed, len(borrowed))
        time.sleep(0.05)
        with lock:
            borrowed.remove(device)
        summary = _write_campaign(
            tmp_path / f"run-{condition}-f{fold:02d}",
            fold_index=fold,
            condition=condition,
        )
        return subprocess.CompletedProcess(command, 0, json.dumps(summary), "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    jobs = runner._build_jobs(
        script_path=Path("scripts/run_hierarchical_scientist.py"),
        config_path=Path("config.yaml"),
        conditions=["kg_base", "kg_base_al"],
        folds=[0, 1],
        seed=11,
        expected_rounds=3,
        expected_budget=16,
        expected_candidate_limit=32,
        output_root=tmp_path / "runs",
        python_executable=sys.executable,
    )
    results = runner.run_fold_jobs(
        jobs,
        max_parallel=4,
        project_dir=tmp_path,
        output_dir=tmp_path / "logs",
        cuda_devices=("0", "1", "2", "3"),
    )
    assert max_borrowed == 4
    assert set(seen) == {"0", "1", "2", "3"}
    assert {result.cuda_device for result in results} == {"0", "1", "2", "3"}
    assert all(result.status == "passed" for result in results)


def test_explicit_cuda_devices_must_cover_max_parallel(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hierarchical_scientist.py",
            "--dry-run",
            "--max-parallel",
            "4",
            "--cuda-devices",
            "0,1",
        ],
    )
    with pytest.raises(SystemExit, match="exceeds 2 CUDA"):
        runner.main()


def test_fold_failure_is_recorded_without_losing_other_folds(tmp_path, monkeypatch) -> None:
    runner = _load_runner()

    def fake_run(command, **_kwargs):
        fold = int(command[command.index("--worker-fold") + 1])
        condition = command[command.index("--worker-condition") + 1]
        if fold == 1:
            return subprocess.CompletedProcess(command, 9, "", "simulated failure")
        summary = _write_campaign(tmp_path / f"run-f{fold:02d}", fold_index=fold, condition=condition)
        return subprocess.CompletedProcess(command, 0, json.dumps(summary), "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    jobs = runner._build_jobs(
        script_path=Path("scripts/run_hierarchical_scientist.py"),
        config_path=Path("config.yaml"),
        conditions=["hierarchical"],
        folds=[0, 1, 2],
        seed=11,
        expected_rounds=3,
        expected_budget=16,
        expected_candidate_limit=32,
        output_root=tmp_path / "runs",
        python_executable=sys.executable,
    )
    results = runner.run_fold_jobs(
        jobs,
        max_parallel=3,
        project_dir=tmp_path,
        output_dir=tmp_path / "logs",
    )
    assert [result.status for result in results] == ["passed", "failed", "passed"]
    assert results[1].returncode == 9
    assert "simulated failure" in Path(results[1].stderr_log).read_text(encoding="utf-8")
    integrity = runner.paired_fold_integrity(
        results, folds=[0, 1, 2], conditions=["hierarchical"]
    )
    assert [item["passed"] for item in integrity] == [True, False, True]


def test_audit_rejects_campaign_missing_succeeded_hypothesis_pipeline(tmp_path) -> None:
    runner = _load_runner()
    summary = _write_campaign(
        tmp_path / "run-missing-pipeline",
        fold_index=0,
        write_pipeline=False,
    )
    audit = runner.audit_hierarchical_run(
        summary,
        condition="hierarchical",
        expected_fold=0,
        expected_rounds=3,
        expected_budget=16,
        expected_candidate_limit=32,
    )
    assert audit["passed"] is False
    assert "pipeline_present_for_every_completed_round" in audit["failed_checks"]


def test_audit_accepts_three_succeeded_channel_branches(tmp_path) -> None:
    runner = _load_runner()
    summary = _write_campaign(tmp_path / "run-ok", fold_index=2)
    audit = runner.audit_hierarchical_run(
        summary,
        condition="hierarchical",
        expected_fold=2,
        expected_rounds=3,
        expected_budget=16,
        expected_candidate_limit=32,
    )
    assert audit["passed"] is True
    assert audit["failed_checks"] == []


def test_apply_condition_can_disable_hierarchy_for_single_agent_ablation() -> None:
    runner = _load_runner()
    base = load_experiment_config("configs/experiments/hierarchical_scientist.deepseek.yaml")
    hierarchical = runner.apply_condition(
        base, "hierarchical", fold=1, seed=11, output_root=Path("artifacts/runs")
    )
    single = runner.apply_condition(
        base, "single", fold=1, seed=11, output_root=Path("artifacts/runs")
    )
    assert hierarchical.hierarchical_hypothesis.enabled is True
    assert single.hierarchical_hypothesis.enabled is False
    assert hierarchical.task.fold_index == 1
    assert single.condition == "single"
    assert hierarchical.run_label.endswith("-f01")
    assert hierarchical.knowledge.physchem is True
    assert single.hierarchical_hypothesis.enabled is False


def test_apply_condition_configures_base_kg_modes_without_feature_tools(tmp_path) -> None:
    runner = _load_runner()
    base = load_experiment_config("configs/experiments/hierarchical_scientist.deepseek.yaml")
    output_root = tmp_path / "runs"
    kg_base = runner.apply_condition(base, "kg_base", fold=0, seed=11, output_root=output_root)
    rag = runner.apply_condition(base, "kg_base_rag", fold=1, seed=11, output_root=output_root)
    al = runner.apply_condition(base, "kg_base_al", fold=2, seed=11, output_root=output_root)
    assert kg_base.hierarchical_hypothesis.enabled is False
    assert kg_base.knowledge.physchem is False
    assert kg_base.knowledge.conservation is False
    assert kg_base.knowledge.structure is False
    assert kg_base.knowledge.kg is True
    assert kg_base.knowledge.local_knowledge.enabled is False
    assert kg_base.kg_interaction.feature_tool_strategy == "context_only"
    assert not set(kg_base.kg_interaction.enabled_operators).intersection(runner.FEATURE_OPERATORS)
    assert rag.knowledge.local_knowledge.enabled is True
    assert rag.knowledge.local_knowledge.allow_remote_context is True
    assert "gb1-reasoning-routes-qwen-v4.sqlite" in str(
        rag.knowledge.local_knowledge.index_path
    )
    assert "kg_base_rag-f01-overlay.sqlite" in str(
        rag.knowledge.local_knowledge.retrieval_overlay_path
    )
    assert "query_local_knowledge" in rag.kg_interaction.enabled_operators
    assert rag.kg_interaction.feature_tool_strategy == "context_only"
    assert al.active_learning.enabled is True
    assert al.generation.selection_driver == "active_learning"
    assert al.generation.quota_allocation.enabled is False
    assert al.knowledge.local_knowledge.enabled is False
    corpus_paths = {
        str(rag.knowledge.local_knowledge.corpus_index_path),
        str(
            runner.apply_condition(
                base, "kg_base_rag", fold=2, seed=11, output_root=output_root
            ).knowledge.local_knowledge.corpus_index_path
        ),
    }
    assert len(corpus_paths) == 1
    overlay_paths = {
        str(rag.knowledge.local_knowledge.retrieval_overlay_path),
        str(
            runner.apply_condition(
                base, "kg_base_rag", fold=2, seed=11, output_root=output_root
            ).knowledge.local_knowledge.retrieval_overlay_path
        ),
    }
    assert len(overlay_paths) == 2
    features_rag = runner.apply_condition(
        base, "kg_3features_rag", fold=0, seed=11, output_root=output_root
    )
    assert features_rag.hierarchical_hypothesis.enabled is True
    assert features_rag.knowledge.physchem is True
    assert features_rag.knowledge.local_knowledge.enabled is True
    assert "query_local_knowledge" in features_rag.kg_interaction.enabled_operators
    assert set(features_rag.kg_interaction.enabled_operators).intersection(runner.FEATURE_OPERATORS)
    assert features_rag.kg_interaction.required_tool_calls(include_rag=True) == 15
    assert features_rag.kg_interaction.max_tool_calls == 18
    with pytest.raises(ValueError, match="complete runtime plan"):
        replace(
            features_rag,
            kg_interaction=replace(features_rag.kg_interaction, max_tool_calls=14),
        )


def test_default_matrix_is_twelve_jobs_in_three_waves_of_four(
    tmp_path, monkeypatch, capsys
) -> None:
    runner = _load_runner()
    split_root = tmp_path / "split"
    split_root.mkdir()
    (split_root / "manifest.public.json").write_text(
        json.dumps(
            {
                "n_folds": 5,
                "strategy": "al96_closed_loop",
                "protocol_version": "GB1-AL96-5CV-v1",
            }
        ),
        encoding="utf-8",
    )
    base = load_experiment_config("configs/experiments/hierarchical_scientist.deepseek.yaml")
    real_load = runner.load_experiment_config

    def fake_load(path, overrides=None):
        if Path(path).name == "hierarchical_scientist.deepseek.yaml":
            return replace(base, task=replace(base.task, split_root=split_root))
        return real_load(path, overrides=overrides)

    monkeypatch.setattr(runner, "load_experiment_config", fake_load)
    monkeypatch.setattr(sys, "argv", ["run_hierarchical_scientist.py", "--dry-run"])
    runner.main()
    schedule = json.loads(capsys.readouterr().out)
    assert schedule["conditions"] == list(runner.DEFAULT_CONDITIONS)
    assert schedule["folds"] == [0, 1, 2]
    assert schedule["max_parallel"] == 4
    assert schedule["cuda_assignment"]["policy"] == "auto"
    assert schedule["cuda_assignment"]["isolation"] == "CUDA_VISIBLE_DEVICES"
    assert schedule["cuda_assignment"]["worker_kermut_device"] == "cuda:0"
    assert schedule["expected_waves"] == 3
    assert schedule["batch_review_scope"] == {
        "controls": True,
        "diversity": True,
    }
    assert schedule["placeholder_predictor"] is False
    assert schedule["max_tokens"] == 32768
    assert schedule["fitness_predictor"] == "kermut"
    assert schedule["use_fitness_predictors"] is False
    assert schedule["predictor_roles"] == {
        "agent_uq_acquisition": "disabled",
        "active_learning_acquisition": "kermut",
        "post_selection_dry_validation": "kermut",
        "oracle_measurement": "approved_batch_only",
    }
    assert schedule["rag_backend"]["embedding"] == "text-embedding-v4"
    assert schedule["rag_backend"]["reranker"] == "qwen3-rerank"
    assert schedule["kg_tool_call_budget"]["kg_3features_rag"] == {
        "required": 15,
        "configured": 18,
    }
    jobs = schedule["jobs"]
    assert len(jobs) == 12
    assert [job["condition"] for job in jobs] == [
        "kg_base",
        "kg_base_rag",
        "kg_base_al",
        "kg_3features_rag",
        "kg_base",
        "kg_base_rag",
        "kg_base_al",
        "kg_3features_rag",
        "kg_base",
        "kg_base_rag",
        "kg_base_al",
        "kg_3features_rag",
    ]
    assert [job["fold_index"] for job in jobs] == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
    ]
    assert {job["condition"] for job in jobs} == set(runner.DEFAULT_CONDITIONS)
    assert {job["fold_index"] for job in jobs} == {0, 1, 2}
    assert all("--worker-placeholder-predictor" not in job["command"] for job in jobs)
    assert all("--worker-max-tokens" in job["command"] for job in jobs)
    assert all(
            job["command"][job["command"].index("--worker-max-tokens") + 1] == "32768"
        for job in jobs
    )


def test_batch_review_scope_flags_are_audited_and_forwarded(
    tmp_path, monkeypatch, capsys
) -> None:
    runner = _load_runner()
    split_root = tmp_path / "split"
    split_root.mkdir()
    (split_root / "manifest.public.json").write_text(
        json.dumps(
            {
                "n_folds": 5,
                "strategy": "al96_closed_loop",
                "protocol_version": "GB1-AL96-5CV-v1",
            }
        ),
        encoding="utf-8",
    )
    base = load_experiment_config(
        "configs/experiments/hierarchical_scientist.deepseek.yaml"
    )
    real_load = runner.load_experiment_config

    def fake_load(path, overrides=None):
        if Path(path).name == "hierarchical_scientist.deepseek.yaml":
            return replace(base, task=replace(base.task, split_root=split_root))
        return real_load(path, overrides=overrides)

    monkeypatch.setattr(runner, "load_experiment_config", fake_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hierarchical_scientist.py",
            "--dry-run",
            "--disable-batch-control-review",
            "--disable-batch-diversity-review",
        ],
    )
    runner.main()
    schedule = json.loads(capsys.readouterr().out)
    assert schedule["batch_review_scope"] == {
        "controls": False,
        "diversity": False,
    }
    assert all(
        "--disable-batch-control-review" in job["command"]
        and "--disable-batch-diversity-review" in job["command"]
        for job in schedule["jobs"]
    )


def test_six_way_parallelism_runs_a_full_wave(tmp_path, monkeypatch) -> None:
    runner = _load_runner()
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_run(command, **_kwargs):
        nonlocal active, max_active
        fold = int(command[command.index("--worker-fold") + 1])
        condition = command[command.index("--worker-condition") + 1]
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        summary = _write_campaign(
            tmp_path / f"run-{condition}-f{fold:02d}",
            fold_index=fold,
            condition=condition,
            write_pipeline=runner.CONDITION_SPECS[condition].hierarchical,
        )
        return subprocess.CompletedProcess(command, 0, json.dumps(summary), "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    jobs = runner._build_jobs(
        script_path=Path("scripts/run_hierarchical_scientist.py"),
        config_path=Path("config.yaml"),
        conditions=list(runner.DEFAULT_CONDITIONS),
        folds=[0, 1, 2],
        seed=11,
        expected_rounds=3,
        expected_budget=16,
        expected_candidate_limit=32,
        output_root=tmp_path / "runs",
        python_executable=sys.executable,
    )
    results = runner.run_fold_jobs(
        jobs,
        max_parallel=6,
        project_dir=tmp_path,
        output_dir=tmp_path / "logs",
    )
    assert len(jobs) == 12
    assert max_active == 6
    assert len(results) == 12
    assert all(result.status == "passed" for result in results)


def test_audit_accepts_base_kg_without_feature_pipeline(tmp_path) -> None:
    runner = _load_runner()
    summary = _write_campaign(
        tmp_path / "run-kg-base",
        fold_index=0,
        condition="kg_base",
        write_pipeline=False,
    )
    audit = runner.audit_hierarchical_run(
        summary,
        condition="kg_base",
        expected_fold=0,
        expected_rounds=3,
        expected_budget=16,
        expected_candidate_limit=32,
    )
    assert audit["passed"] is True
    assert audit["failed_checks"] == []


def test_placeholder_predictor_flag_is_rejected(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_hierarchical_scientist.py", "--placeholder-predictor", "--dry-run"],
    )
    with pytest.raises(SystemExit, match="cannot use --placeholder-predictor"):
        runner.main()


def test_apply_token_budget_sets_all_role_limits() -> None:
    runner = _load_runner()
    base = load_experiment_config("configs/experiments/hierarchical_scientist.deepseek.yaml")
    updated = runner.apply_token_budget(base, 20000)
    assert updated.llm.max_tokens == 20000
    assert updated.critic.max_tokens == 20000
    assert updated.hierarchical_hypothesis.child_max_tokens == 20000
    assert updated.hierarchical_hypothesis.child_critic_max_tokens == 20000
    assert updated.hierarchical_hypothesis.main_critic_max_tokens == 20000
    with pytest.raises(ValueError, match="between 1 and 131072"):
        runner.apply_token_budget(base, 131073)


def test_validate_formal_fitness_configuration_rejects_generation_predictor() -> None:
    runner = _load_runner()
    base = load_experiment_config("configs/experiments/hierarchical_scientist.deepseek.yaml")
    assert runner.validate_formal_fitness_configuration(base) is base
    invalid = replace(
        base,
        generation=replace(
            base.generation,
            use_fitness_predictors=True,
            predictor_models=(base.model,),
            predictor_weight=1.0,
        ),
    )
    with pytest.raises(ValueError, match="must not mix fitness predictions"):
        runner.validate_formal_fitness_configuration(invalid)


def test_audit_rejects_placeholder_predictor(tmp_path) -> None:
    runner = _load_runner()
    summary = _write_campaign(tmp_path / "run-placeholder", fold_index=0)
    summary["placeholder_predictor"] = True
    audit = runner.audit_hierarchical_run(
        summary,
        condition="hierarchical",
        expected_fold=0,
        expected_rounds=3,
        expected_budget=16,
        expected_candidate_limit=32,
    )
    assert audit["passed"] is False
    assert "placeholder_predictor_disabled" in audit["failed_checks"]
