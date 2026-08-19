import json

from scripts.audit_agent_completion import audit_run


def test_completion_audit_fails_closed_for_missing_or_ineligible_run(tmp_path) -> None:
    assert audit_run(tmp_path)["passed"] is False

    (tmp_path / "completion_manifest.json").write_text(
        json.dumps(
            {
                "artifact_finalized": True,
                "run_status": "failed",
                "experiment_status": "failed",
                "evaluation_status": "not_evaluated",
                "pass_eligible": False,
                "expected_rounds": 2,
                "completed_rounds": 1,
                "aborted_rounds": 1,
                "required_node_failures": ["round_2:child_failed"],
                "fallback_nodes": [],
            }
        ),
        encoding="utf-8",
    )
    result = audit_run(tmp_path)
    assert result["passed"] is False
    assert "not eligible" in result["errors"][0]
