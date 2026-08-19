from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from fitness_agents.agents.structured_completion import complete_structured


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: str


class _Client:
    base_url = "https://api.deepseek.com"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("Message", (), {"content": '{"verdict":"APPROVE"}'})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type("Response", (), {"choices": [choice], "usage": None})()


def test_reasoning_draft_and_json_render_are_separate_requests() -> None:
    client = _Client()
    output = complete_structured(
        client=client,
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Review."},
        ],
        output_type=_Output,
        reasoning_effort="high",
        thinking="enabled",
        separate_json_render=True,
        transport_retries=0,
        truncation_retries=0,
        syntax_retries=0,
        schema_retries=0,
        semantic_retries=0,
    )
    assert output.verdict == "APPROVE"
    assert len(client.calls) == 2
    assert client.calls[0]["reasoning_effort"] == "high"
    assert client.calls[0]["extra_body"]["thinking"]["type"] == "enabled"
    assert "reasoning_effort" not in client.calls[1]
    assert client.calls[1]["extra_body"]["thinking"]["type"] == "disabled"
    assert client.calls[1]["messages"][-2]["role"] == "assistant"


class _BoundedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str = Field(max_length=400)


class _RenderClient(_Client):
    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = (
            '{"statement":"' + ("x" * 411) + '"}'
            if len(self.calls) == 1
            else '{"statement":"compressed"}'
        )
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type("Response", (), {"choices": [choice], "usage": None})()


def test_first_json_render_receives_generated_string_limits() -> None:
    client = _RenderClient()
    output = complete_structured(
        client=client,
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Draft a statement."},
        ],
        output_type=_BoundedOutput,
        thinking="enabled",
        separate_json_render=True,
        transport_retries=0,
        truncation_retries=0,
        syntax_retries=0,
        schema_retries=0,
        semantic_retries=0,
    )
    assert output.statement == "compressed"
    render_instruction = client.calls[1]["messages"][-1]["content"]
    assert "statement: max 400 characters" in render_instruction
    assert "MAY compress only free-text prose" in render_instruction
