from __future__ import annotations

import asyncio

import pytest
from agents import AgentOutputSchema

from fitness_agents.agents.output_contracts import HypothesisOutput, ReThinkOutput
from fitness_agents.agents.sdk_agents import (
    AgentsSDKScientistLLMClient,
    _CompletionsProxy,
    build_scientist_kg_tools,
)


def _sdk_output() -> HypothesisOutput:
    return HypothesisOutput.model_validate(
        {
            "hypothesis_id": "hyp:run:r1",
            "statement": "Visible evidence supports a bounded hypothesis.",
            "preferred_residues": {
                "39": ["W"],
                "40": ["D"],
                "41": ["G"],
                "54": ["V"],
            },
            "evidence_ids": [],
            "expected_outcome": "Enrichment relative to random selection.",
            "falsification_criterion": "Revise if wet validation does not improve.",
            "parent_hypothesis_id": None,
        }
    )


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        output = _sdk_output()
        kwargs["validate_output"](output)
        return output


def test_sdk_scientist_receives_only_explicit_sanitized_payload() -> None:
    runtime = _FakeRuntime()
    client = AgentsSDKScientistLLMClient(sdk_runtime=runtime)
    context = {
        "run_id": "run",
        "round_id": 1,
        "expected_hypothesis_id": "hyp:run:r1",
        "visible_observations": [],
        "previous_hypothesis_id": None,
    }

    result = client.generate_hypothesis(
        sanitized_context=context,
        evidence=[],
        output_schema={},
        trace_context={"run_id": "run", "round_id": 1, "role": "scientist"},
    )

    assert result.hypothesis_id == "hyp:run:r1"
    assert runtime.calls[0]["input_payload"] == {"context": context, "evidence": []}
    assert runtime.calls[0]["output_type"] is HypothesisOutput


def test_sdk_scientist_rejects_hidden_context_before_runtime() -> None:
    runtime = _FakeRuntime()
    client = AgentsSDKScientistLLMClient(sdk_runtime=runtime)

    with pytest.raises(ValueError, match="Forbidden hidden-label"):
        client.generate_hypothesis(
            sanitized_context={
                "run_id": "run",
                "round_id": 1,
                "expected_hypothesis_id": "hyp:run:r1",
                "oracle_path": "hidden.csv",
            },
            evidence=[],
            output_schema={},
        )

    assert runtime.calls == []


class _ToolSessionStub:
    def call(self, *args, **kwargs):
        del args, kwargs
        return {"ok": True}


def test_sdk_scientist_exposes_only_allowlisted_kg_tools() -> None:
    names = {tool.name for tool in build_scientist_kg_tools(_ToolSessionStub())}

    assert names == {
        "kg_hypothesis_context",
        "kg_explain_variant",
        "kg_compare_variants",
    }
    assert not any(
        forbidden in name
        for name in names
        for forbidden in ("oracle", "final", "experiment", "submit", "backend")
    )


def test_standard_deepseek_tools_keep_local_validation_without_beta_strict() -> None:
    tools = build_scientist_kg_tools(_ToolSessionStub(), strict_mode=False)

    assert all(tool.strict_json_schema is False for tool in tools)


def test_sdk_role_outputs_are_valid_strict_schemas() -> None:
    assert AgentOutputSchema(HypothesisOutput).is_strict_json_schema()
    assert AgentOutputSchema(ReThinkOutput).is_strict_json_schema()


class _AsyncCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return object()


def test_deepseek_proxy_downgrades_json_schema_but_keeps_schema_instruction() -> None:
    delegate = _AsyncCompletions()
    proxy = _CompletionsProxy(delegate)
    asyncio.run(
        proxy.create(
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object", "required": ["key"]}},
            },
            messages=[{"role": "system", "content": "role instructions"}],
        )
    )

    assert delegate.kwargs["response_format"] == {"type": "json_object"}
    assert '"required": ["key"]' in delegate.kwargs["messages"][0]["content"]
