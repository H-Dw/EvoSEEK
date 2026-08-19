from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict, Field

from fitness_agents.agents.output_guards import (
    UnknownEvidenceIdsError,
    classify_output_failure,
    json_salvage,
    retry_instruction,
)
from fitness_agents.agents.remote_llm import (
    RemoteLLMCompletionError,
    complete_json,
    create_openai_client,
)
from fitness_agents.utils.progress import bind_progress, reset_progress


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


def test_openai_sdk_retries_are_disabled_and_timeout_is_explicit(monkeypatch) -> None:
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-secret")
    create_openai_client(provider="deepseek", request_timeout_seconds=45.0)
    assert captured["max_retries"] == 0
    assert captured["timeout"] == 45.0


def test_json_salvage_strips_trailing_commas() -> None:
    repaired = json_salvage('{"ok": true,}')
    assert repaired == {"ok": True}


def test_json_salvage_never_closes_truncated_object() -> None:
    repaired = json_salvage('{"ok": true, "nested": {"a": 1')
    assert repaired is None


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
    assert "reasoning_effort" not in client.calls[1]
    extra = client.calls[1].get("extra_body") or {}
    assert extra.get("thinking", {}).get("type") == "disabled"


def test_thinking_disabled_omits_reasoning_effort_on_first_request() -> None:
    client = _ScriptedClient([('{"ok": true}', "stop")])
    complete_json(
        client=client,
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "json"}],
        reasoning_effort="high",
        thinking="disabled",
    )
    assert "reasoning_effort" not in client.calls[0]


def test_truncated_payload_is_not_replayed_as_a_repair_draft() -> None:
    client = _ScriptedClient(
        [('{"unfinished":', "length"), ('{"ok": true}', "stop")]
    )
    complete_json(
        client=client,
        model="unit-test-model",
        messages=[{"role": "user", "content": "json"}],
    )
    assert not any(
        message["role"] == "assistant" for message in client.calls[1]["messages"]
    )


def test_schema_repair_replays_visible_json_and_generated_constraints() -> None:
    class _Contract(BaseModel):
        model_config = ConfigDict(extra="forbid")
        verdict: str = Field(pattern="^(APPROVE|REVISE)$")
        summary: str = Field(max_length=8)

    calls = 0

    def validator(payload: dict) -> dict:
        nonlocal calls
        calls += 1
        return _Contract.model_validate(payload).model_dump()

    client = _ScriptedClient(
        [
            ('{"verdict":"APPROVE","summary":"too long for contract"}', "stop"),
            ('{"verdict":"APPROVE","summary":"ok"}', "stop"),
        ]
    )
    complete_json(
        client=client,
        model="unit-test-model",
        messages=[{"role": "user", "content": "json"}],
        schema=_Contract.model_json_schema(),
        schema_retries=1,
        validator=validator,
    )
    repair_messages = client.calls[1]["messages"]
    assert repair_messages[-2]["role"] == "assistant"
    assert "too long" in repair_messages[-2]["content"]
    assert "summary" in repair_messages[-1]["content"]
    assert "maxLength" in repair_messages[-1]["content"]


def test_syntax_and_schema_budgets_are_independent() -> None:
    validation_calls = 0

    def validator(payload: dict) -> dict:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            raise ValueError("schema mismatch")
        return payload

    client = _ScriptedClient(
        [
            ('{"bad": true "comma": false}', "stop"),
            ('{"schema": "wrong"}', "stop"),
            ('{"ok": true}', "stop"),
        ]
    )
    payload = complete_json(
        client=client,
        model="unit-test-model",
        messages=[{"role": "user", "content": "json"}],
        syntax_retries=1,
        schema_retries=1,
        validator=validator,
    )
    assert payload == {"ok": True}
    assert len(client.calls) == 3


def test_extraction_rejects_prose_wrapped_or_multiple_objects() -> None:
    for content in ('answer: {"ok": true}', '{"a": 1} {"b": 2}'):
        client = _ScriptedClient([(content, "stop")])
        with pytest.raises(RemoteLLMCompletionError):
            complete_json(
                client=client,
                model="unit-test-model",
                messages=[{"role": "user", "content": "json"}],
                syntax_retries=0,
            )


def test_retry_observability_records_hash_paths_usage_budget_and_disposition() -> None:
    class _Contract(BaseModel):
        model_config = ConfigDict(extra="forbid")
        summary: str = Field(max_length=4)

    class _Capture:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict]] = []

        def heartbeat(self, message, *, log=True, **payload):
            del message, log, payload

        def report(self, event_type, *, message, persist=True, **payload):
            del message, persist
            self.events.append((event_type, payload))

    client = _ScriptedClient(
        [('{"summary":"too long"}', "stop"), ('{"summary":"ok"}', "stop")]
    )
    capture = _Capture()
    token = bind_progress(capture)
    try:
        complete_json(
            client=client,
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "json"}],
            schema=_Contract.model_json_schema(),
            thinking="disabled",
            reasoning_effort="high",
            schema_retries=1,
            validator=lambda value: _Contract.model_validate(value).model_dump(),
            trace_context={"role": "unit_contract"},
        )
    finally:
        reset_progress(token)
    retry = next(payload for name, payload in capture.events if name == "llm_request_retry")
    assert retry["role"] == "unit_contract"
    assert retry["thinking"] == "disabled"
    assert retry["completion_tokens"] == 20
    assert retry["validation_errors"][0]["path"] == "summary"
    assert len(retry["invalid_payload_sha256"]) == 64
    assert retry["retry_budget"]["limits"]["schema"] == 1
    assert retry["disposition"] == "retry"


def test_complete_json_never_accepts_complete_json_with_length_finish_reason() -> None:
    client = _ScriptedClient(
        [
            ('{"apparently_complete": true}', "length"),
            ('{"ok": true}', "stop"),
        ]
    )
    payload = complete_json(
        client=client,
        model="unit-test-model",
        messages=[{"role": "user", "content": "json"}],
        truncation_retries=1,
        transport_retries=0,
    )
    assert payload == {"ok": True}
    assert len(client.calls) == 2


class _HTTPError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"http {status_code}")
        self.status_code = status_code


class _FailingClient(_ScriptedClient):
    def __init__(self, failures: list[Exception], responses: list[tuple[str, str]]) -> None:
        super().__init__(responses)
        self.failures = failures

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        content, finish_reason = self.responses[0]
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
        return type("Response", (), {"choices": [choice], "usage": None})()


def test_non_retryable_http_error_fails_after_one_external_request() -> None:
    client = _FailingClient([_HTTPError(403)], [('{"ok": true}', "stop")])
    try:
        complete_json(
            client=client,
            model="unit-test-model",
            messages=[{"role": "user", "content": "json"}],
            transport_retries=2,
        )
    except RuntimeError as error:
        assert isinstance(error.__cause__, _HTTPError)
        assert isinstance(error, RemoteLLMCompletionError)
        assert error.error_code == "HTTP_403"
        assert error.failure_category == "transport"
        assert error.request_started is True
    else:
        raise AssertionError("403 must fail closed")
    assert len(client.calls) == 1


def test_retryable_http_error_uses_only_transport_budget() -> None:
    client = _FailingClient([_HTTPError(503)], [('{"ok": true}', "stop")])
    payload = complete_json(
        client=client,
        model="unit-test-model",
        messages=[{"role": "user", "content": "json"}],
        transport_retries=1,
    )
    assert payload == {"ok": True}
    assert len(client.calls) == 2


def test_content_filter_and_tool_call_finish_reasons_fail_without_retry() -> None:
    for finish_reason in ("content_filter", "tool_calls"):
        client = _ScriptedClient([("{}", finish_reason)])
        try:
            complete_json(
                client=client,
                model="unit-test-model",
                messages=[{"role": "user", "content": "json"}],
                transport_retries=2,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{finish_reason} must fail closed")
        assert len(client.calls) == 1


def test_insufficient_system_resource_uses_transport_retry() -> None:
    client = _ScriptedClient(
        [("", "insufficient_system_resource"), ('{"ok": true}', "stop")]
    )
    payload = complete_json(
        client=client,
        model="unit-test-model",
        messages=[{"role": "user", "content": "json"}],
        transport_retries=1,
    )
    assert payload == {"ok": True}
    assert len(client.calls) == 2


def test_formal_unknown_evidence_does_not_silently_strip_and_pass() -> None:
    def validator(payload: dict) -> dict:
        del payload
        raise UnknownEvidenceIdsError(
            ["ev:missing"],
            frozenset({"ev:1"}),
            stripped_payload={"ok": True, "evidence_ids": []},
        )

    client = _ScriptedClient([('{"ok": true}', "stop")] * 2)
    try:
        complete_json(
            client=client,
            model="unit-test-model",
            messages=[{"role": "user", "content": "json"}],
            transport_retries=0,
            unknown_evidence_retries=1,
            validator=validator,
            allow_unknown_evidence_stripping=False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("formal citation failure must not be silently stripped")
    assert len(client.calls) == 2


def test_prompt_preflight_budget_fails_before_external_request() -> None:
    client = _ScriptedClient([('{"ok": true}', "stop")])
    try:
        complete_json(
            client=client,
            model="unit-test-model",
            messages=[{"role": "user", "content": "x" * 101}],
            max_input_chars=100,
        )
    except RuntimeError as error:
        assert "prompt" in str(error.__cause__).lower()
        assert isinstance(error, RemoteLLMCompletionError)
        assert error.error_code == "PROMPT_BUDGET_EXCEEDED"
        assert error.failure_category == "budget"
        assert error.input_chars == 101
        assert error.request_started is False
    else:
        raise AssertionError("oversized prompt must fail preflight")
    assert client.calls == []


def test_schema_failure_keeps_structured_terminal_code() -> None:
    client = _ScriptedClient([('{"ok": true}', "stop")])

    def reject_schema(payload: dict) -> dict:
        del payload
        raise ValueError("schema mismatch")

    with pytest.raises(RemoteLLMCompletionError) as captured:
        complete_json(
            client=client,
            model="unit-test-model",
            messages=[{"role": "user", "content": "json"}],
            schema_retries=0,
            validator=reject_schema,
        )

    assert captured.value.error_code == "OUTPUT_SCHEMA_INVALID"
    assert captured.value.failure_category == "output"
    assert captured.value.request_started is True


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


def test_unknown_evidence_is_never_rewritten_after_retries_exhausted() -> None:
    def validator(payload: dict) -> dict:
        del payload
        raise UnknownEvidenceIdsError(
            ["ev:missing"],
            frozenset({"ev:1"}),
            stripped_payload={"ok": True, "evidence_ids": []},
        )

    client = _ScriptedClient([('{"ok": true}', "stop")] * 3)
    with pytest.raises(RemoteLLMCompletionError) as captured:
        complete_json(
            client=client,
            model="unit-test-model",
            messages=[{"role": "user", "content": "json"}],
            unknown_evidence_retries=2,
            validator=validator,
            allow_unknown_evidence_stripping=True,
        )
    assert captured.value.error_code == "OUTPUT_EVIDENCE_IDS_INVALID"
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
