from __future__ import annotations

from pydantic import ValidationError

from fitness_agents.agents.output_guards import SemanticOutputValidationError
from fitness_agents.agents.remote_llm import RemoteLLMCompletionError
from fitness_agents.loop.orchestrator import is_hypothesis_generation_error
from fitness_agents.loop.review import HypothesisGenerationFailed


def test_is_hypothesis_generation_error_covers_scientist_json_failures() -> None:
    assert is_hypothesis_generation_error(
        SemanticOutputValidationError("soft hypothesis prose")
    )
    assert is_hypothesis_generation_error(
        RemoteLLMCompletionError(
            "OUTPUT_SEMANTIC_INVALID",
            failure_category="output",
            input_chars=12,
            request_started=True,
            detail="semantic retries exhausted",
        )
    )
    assert is_hypothesis_generation_error(ValidationError.from_exception_data("Model", []))
    assert not is_hypothesis_generation_error(RuntimeError("disk full"))


def test_hypothesis_generation_failed_is_round_abort() -> None:
    error = HypothesisGenerationFailed(SemanticOutputValidationError("boom"))
    assert error.code == "HYPOTHESIS_NODE_FAILED"
    assert error.terminal_policy == "abort_round"
    assert "SemanticOutputValidationError" in str(error)
