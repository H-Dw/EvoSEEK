import pytest
from pydantic import ValidationError

from fitness_agents.contracts.hypothesis_pipeline import CompletionManifest


def test_complete_run_is_eligible_but_not_automatically_passed() -> None:
    manifest = CompletionManifest(
        artifact_finalized=True,
        run_status="completed",
        experiment_status="completed",
        evaluation_status="eligible",
        pass_eligible=True,
        expected_rounds=3,
        completed_rounds=3,
        aborted_rounds=0,
    )
    assert manifest.pass_eligible is True
    assert manifest.evaluation_status == "eligible"


def test_incomplete_or_fallback_run_cannot_be_passed() -> None:
    with pytest.raises(ValidationError, match="incomplete or degraded"):
        CompletionManifest(
            artifact_finalized=True,
            run_status="failed",
            experiment_status="partial",
            evaluation_status="passed",
            pass_eligible=False,
            expected_rounds=3,
            completed_rounds=2,
            aborted_rounds=1,
            required_node_failures=("round_3:MAIN_CRITIC_NOT_APPROVED",),
        )

    fallback = CompletionManifest(
        artifact_finalized=True,
        run_status="completed",
        experiment_status="completed",
        evaluation_status="not_evaluated",
        pass_eligible=False,
        expected_rounds=1,
        completed_rounds=1,
        aborted_rounds=0,
        fallback_nodes=("round_1:rethink",),
    )
    assert fallback.pass_eligible is False
