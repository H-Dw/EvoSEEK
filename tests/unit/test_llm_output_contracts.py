from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from fitness_agents.agents.llm import HYPOTHESIS_SCHEMA, OpenAICompatibleLLMClient
from fitness_agents.agents.output_contracts import HypothesisOutput


class _SequenceClient:
    base_url = "https://example.invalid/v1"

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.calls = 0
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        del kwargs
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        message = type("Message", (), {"content": json.dumps(payload)})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type("Response", (), {"choices": [choice], "usage": None})()


def _valid_payload() -> dict:
    return {
        "hypothesis_id": "hyp:run:r1",
        "statement": "Visible evidence supports a bounded four-site hypothesis.",
        "preferred_residues": {
            "39": ["W"],
            "40": ["D"],
            "41": ["G"],
            "54": ["V"],
        },
        "evidence_ids": [],
        "expected_outcome": "Enrichment relative to random selection.",
        "falsification_criterion": "Revise if the wet batch median does not improve.",
        "parent_hypothesis_id": None,
    }


def _client(remote: _SequenceClient) -> OpenAICompatibleLLMClient:
    client = OpenAICompatibleLLMClient.__new__(OpenAICompatibleLLMClient)
    client.model = "unit-test-model"
    client.temperature = 0.0
    client.max_tokens = 1024
    client.reasoning_effort = None
    client.thinking = None
    client.client = remote
    return client


def _context() -> dict:
    return {
        "run_id": "run",
        "round_id": 1,
        "expected_hypothesis_id": "hyp:run:r1",
        "visible_observations": [],
        "previous_hypothesis_id": None,
    }


def test_missing_hypothesis_id_retries_inside_json_boundary() -> None:
    missing_id = _valid_payload()
    del missing_id["hypothesis_id"]
    remote = _SequenceClient([missing_id, _valid_payload()])

    hypothesis = _client(remote).generate_hypothesis(
        sanitized_context=_context(),
        evidence=[],
        output_schema=HYPOTHESIS_SCHEMA,
    )

    assert remote.calls == 2
    assert hypothesis.hypothesis_id == "hyp:run:r1"


def test_exhausted_missing_key_is_validation_error_not_key_error() -> None:
    missing_id = _valid_payload()
    del missing_id["hypothesis_id"]
    remote = _SequenceClient([missing_id])

    with pytest.raises(RuntimeError) as captured:
        _client(remote).generate_hypothesis(
            sanitized_context=_context(),
            evidence=[],
            output_schema=HYPOTHESIS_SCHEMA,
        )

    assert remote.calls == 3
    assert isinstance(captured.value.__cause__, ValidationError)


def test_hypothesis_output_schema_has_fixed_site_keys() -> None:
    schema = HypothesisOutput.model_json_schema()
    preferred_ref = schema["properties"]["preferred_residues"]["$ref"]
    preferred_name = preferred_ref.rsplit("/", 1)[-1]
    site_schema = schema["$defs"][preferred_name]

    assert site_schema["additionalProperties"] is False
    assert set(site_schema["required"]) == {"39", "40", "41", "54"}
