from __future__ import annotations

import json

from fitness_agents.agents.output_guards import (
    UnknownEvidenceIdsError,
    classify_output_failure,
    json_salvage,
    retry_instruction,
)
from fitness_agents.agents.remote_llm import complete_json


class _ScriptedClient:
    base_url = "https://example.invalid/v1"

    def __init__(self, responses: list[tuple[str, str]]) -> None:
        self.responses = responses
        self.calls: list[dict] = []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content, finish_reason = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
        usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "completion_tokens_details": None,
            },
        )()
        return type("Response", (), {"choices": [choice], "usage": usage})()


def test_json_salvage_strips_trailing_commas() -> None:
    repaired = json_salvage('{"ok": true,}')
    assert repaired == {"ok": True}


def test_json_salvage_closes_truncated_object() -> None:
    repaired = json_salvage('{"ok": true, "nested": {"a": 1')
    assert repaired is not None
    assert repaired["ok"] is True


def test_complete_json_salvages_trailing_comma_without_retry() -> None:
    client = _ScriptedClient([('{"ok": true,}', "stop")])
    payload = complete_json(
        client=client,
        model="unit-test-model",
        messages=[{"role": "user", "content": "json"}],
    )
    assert payload == {"ok": True}
    assert len(client.calls) == 1


def test_complete_json_retries_truncated_json_and_raises_budget() -> None:
    client = _ScriptedClient(
        [
            ('{"statement": "incomplete', "length"),
            ('{"ok": true}', "stop"),
        ]
    )
    payload = complete_json(
        client=client,
        model="unit-test-model",
        messages=[{"role": "user", "content": "json"}],
        max_tokens=1024,
    )
    assert payload == {"ok": True}
    assert len(client.calls) == 2
    assert client.calls[0]["max_tokens"] == 1024
    assert client.calls[1]["max_tokens"] > 1024
    retry_text = client.calls[1]["messages"][-1]["content"]
    assert "truncated" in retry_text
    assert "finish_reason=length" in retry_text


def test_complete_json_truncation_lowers_reasoning_effort() -> None:
    client = _ScriptedClient(
        [
            ('{"statement": "incomplete', "length"),
            ('{"ok": true}', "stop"),
        ]
    )
    payload = complete_json(
        client=client,
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "json"}],
        max_tokens=1024,
        reasoning_effort="high",
        thinking="enabled",
    )
    assert payload == {"ok": True}
    assert client.calls[0]["reasoning_effort"] == "high"
    assert client.calls[1]["reasoning_effort"] == "low"
    extra = client.calls[1].get("extra_body") or {}
    assert extra.get("thinking", {}).get("type") == "disabled"


def test_complete_json_retry_mentions_json_decode_position() -> None:
    client = _ScriptedClient(
        [
            ('{"ok": true "broken": 1}', "stop"),
            ('{"ok": true}', "stop"),
        ]
    )
    payload = complete_json(
        client=client,
        model="unit-test-model",
        messages=[{"role": "user", "content": "json"}],
    )
    assert payload == {"ok": True}
    retry_text = client.calls[1]["messages"][-1]["content"]
    assert "character" in retry_text or "JSONDecodeError" in retry_text or "syntax" in retry_text


def test_unknown_evidence_is_stripped_after_retries_exhausted() -> None:
    def validator(payload: dict) -> dict:
        del payload
        raise UnknownEvidenceIdsError(
            ["ev:missing"],
            frozenset({"ev:1"}),
            stripped_payload={"ok": True, "evidence_ids": []},
        )

    client = _ScriptedClient([('{"ok": true}', "stop")] * 3)
    payload = complete_json(
        client=client,
        model="unit-test-model",
        messages=[{"role": "user", "content": "json"}],
        retries=2,
        validator=validator,
    )
    assert payload == {"ok": True, "evidence_ids": []}
    assert len(client.calls) == 3
    assert "ev:missing" in client.calls[1]["messages"][-1]["content"]


def test_classify_truncated_unbalanced_json() -> None:
    failure = classify_output_failure(
        json.JSONDecodeError("Expecting", '{"a":', 5),
        finish_reason="stop",
        content='{"a":',
    )
    assert failure.kind == "truncated"


def test_retry_instruction_includes_allowed_ids() -> None:
    error = UnknownEvidenceIdsError(["ev:x"], frozenset({"ev:1", "ev:2"}))
    failure = classify_output_failure(error, finish_reason=None, content="{}")
    text = retry_instruction(failure, error=error)
    assert "ev:x" in text
    assert "ev:1" in text
