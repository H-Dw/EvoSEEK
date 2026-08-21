from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from fitness_agents.interaction import (
    DeterministicEvolutionIntentParser,
    EvolutionApplicationService,
)
from fitness_agents.interaction.gradio_app import preview_callback, run_callback


def _write_config(tmp_path: Path) -> Path:
    initial_path = tmp_path / "initial.csv"
    pd.DataFrame(
        [
            {"variant_id": f"v{index}", "variant": sequence, "fitness": fitness}
            for index, (sequence, fitness) in enumerate(
                (
                    ("ACDE", 0.0),
                    ("CCDE", 0.1),
                    ("ADDE", 0.2),
                    ("ACEF", 0.3),
                    ("ACNE", 0.4),
                    ("WCDE", 0.5),
                )
            )
        ]
    ).to_csv(initial_path, index=False)
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "task_id": "tiny_open",
                "protein_id": "tiny_open",
                "assay_id": "tiny_assay",
                "wild_type_sites": "CD",
                "mutable_positions": [2, 3],
                "objective": "提高目标结合能力",
                "initial_observations_path": str(initial_path),
                "reference_sequence": "ACDE",
                "sequence_position_offset": 1,
                "numbering_scheme": "one_based",
            }
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        yaml.safe_dump(
            {
                "name": "onehot_heterogeneous_ensemble",
                "feature_provider": "full_sequence_onehot",
                "ridge_members": 2,
                "extra_trees_estimators": 12,
                "capabilities": {
                    "supports_full_sequence": True,
                    "supports_generated_sequences": True,
                },
            }
        ),
        encoding="utf-8",
    )
    knowledge_path = tmp_path / "knowledge.yaml"
    knowledge_path.write_text(
        yaml.safe_dump(
            {"physchem": False, "conservation": False, "structure": False, "kg": False}
        ),
        encoding="utf-8",
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "mode": "knowledge_agent",
                "seed": 7,
                "rounds": 1,
                "budget_per_round": 8,
                "candidate_limit": 0,
                "acquisition": "greedy",
                "ucb_beta": 1.0,
                "diversity_lambda": 0.1,
                "task_config": str(task_path),
                "model_config": str(model_path),
                "knowledge_config": str(knowledge_path),
                "output_root": str(tmp_path / "runs"),
                "llm_provider": "mock",
                "knowledge_enabled": False,
                "designer": {
                    "space": "open_design",
                    "position_policy": "all",
                    "mutation_depth": 1,
                },
                "generation": {"selection_driver": "active_learning"},
                "active_learning": {
                    "enabled": True,
                    "posterior": {
                        "predictor_models": [str(model_path)],
                        "min_training_size": 4,
                        "min_calibration_size": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return experiment_path


def test_intent_parser_understands_all_include_and_exclude() -> None:
    parser = DeterministicEvolutionIntentParser()
    all_intent = parser.parse("希望对 ACDEFG 进行定向进化，提高活性，开放全部位置")
    include_intent = parser.parse("仅优化位置 2, 5, 17", configured_reference="ACDEFG")
    exclude_intent = parser.parse("除位置 2、5 外全部开放", configured_reference="ACDEFG")

    assert all_intent.constraints.position_policy == "all"
    assert all_intent.reference_sequence == "ACDEFG"
    assert include_intent.constraints.position_policy == "include"
    assert include_intent.constraints.include_positions == (2, 5, 17)
    assert exclude_intent.constraints.position_policy == "all_except"
    assert exclude_intent.constraints.exclude_positions == (2, 5)


def test_preview_does_not_create_run_and_reuses_confirmed_design_space(tmp_path: Path) -> None:
    service = EvolutionApplicationService(_write_config(tmp_path))
    message, preview_payload, preview_id = preview_callback(
        service,
        "希望提高结合能力，开放全部位置，输出 8 条",
        "ACDE",
    )

    assert "可确认运行" in message
    assert preview_payload["resolved_positions"] == [1, 2, 3, 4]
    assert preview_payload["generated_candidate_count"] == 76
    assert preview_payload["budget"] == 8
    assert not (tmp_path / "runs").exists()

    status, summary, artifacts = run_callback(service, preview_id, True)

    assert "通过 hard validation" in status
    assert f"run_id: {summary['run_id']}" in status
    assert summary["resolved_positions"] == [1, 2, 3, 4]
    assert summary["critic_verdict"] == "APPROVE"
    assert summary["hard_validation_blockers"] == 0
    assert summary["selected_count"] == 8
    assert any(path.endswith("selected_candidates.fasta") for path in artifacts)
    approved = Path(summary["run_dir"]) / "review" / "approved_batch.json"
    assert json.loads(approved.read_text(encoding="utf-8"))["candidate_ids"]

    include_preview = service.preview("仅优化位置 2, 4，输出 2 条", sequence_text="ACDE")
    assert include_preview.resolved_positions == (2, 4)
    assert include_preview.preview_id != preview_id
    include_result = service.run(include_preview.preview_id, confirmed=True)
    include_dir = Path(str(include_result.summary["run_dir"]))
    design_space = json.loads(
        (include_dir / "design_space.json").read_text(encoding="utf-8")
    )
    hypothesis = json.loads(
        (include_dir / "hypothesis.json").read_text(encoding="utf-8")
    )
    assert design_space["positions"] == [2, 4]
    assert set(map(int, hypothesis["preferred_residues"])) <= {2, 4}


def test_failed_run_keeps_preview_and_exposes_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    service = EvolutionApplicationService(_write_config(tmp_path))
    preview = service.preview("希望提高结合能力，开放全部位置，输出 8 条", sequence_text="ACDE")
    assert preview.ready_for_confirmation

    from fitness_agents.loop.open_design import OpenDesignRunner

    def _fail(self) -> None:
        self.writer.report(
            "open_design_failed",
            message="open design failed",
            phase="failed",
            round_id=1,
            error_type="RuntimeError",
            error_message="boom",
        )
        raise RuntimeError("boom")

    monkeypatch.setattr(OpenDesignRunner, "run", _fail)
    status, summary, artifacts = run_callback(service, preview.preview_id, True)

    assert "运行失败" in status
    assert "boom" in status
    assert "运行目录" in status
    assert summary == {}
    assert any(path.endswith("status.json") for path in artifacts)
    assert any(path.endswith("trace.jsonl") for path in artifacts)

    monkeypatch.undo()
    status, summary, _ = run_callback(service, preview.preview_id, True)
    assert "通过 hard validation" in status
    assert summary["critic_verdict"] == "APPROVE"


def test_preview_blocks_reference_mismatch_and_unconfirmed_submit(tmp_path: Path) -> None:
    service = EvolutionApplicationService(_write_config(tmp_path))
    preview = service.preview("提高活性", sequence_text="ACDF")

    assert preview.ready_for_confirmation is False
    assert any("不一致" in item for item in preview.blockers)
    status, _, _ = run_callback(service, preview.preview_id, False)
    assert "必须确认" in status
    assert not (tmp_path / "runs").exists()
