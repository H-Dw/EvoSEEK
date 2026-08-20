from __future__ import annotations

import json
import threading
import time

from fitness_agents.agents import rethink as rethink_module
from fitness_agents.agents.rethink import NativeReThinkClient
from fitness_agents.contracts.agent_io import ReThinkContextInput


def _reflection(candidate: dict) -> dict:
    return {
        "variant_id": candidate["variant_id"],
        "candidate_relation": "mixed",
        "summary": "Wet and dry evidence remain mixed.",
        "positive_findings": ["A bounded positive signal was observed."],
        "negative_findings": ["Uncertainty remains material."],
        "revised_reason": "Retain the rationale only as round-specific evidence.",
        "next_round_advice": "Test a matched alternative in the next round.",
        "next_round_action": "test_matched_alternative",
        "dimension_assessments": [
            {
                "dimension": dimension,
                "evidence_status": "missing",
                "finding": "No additional dimension evidence is visible.",
                "implication": "Keep the interpretation bounded.",
            }
            for dimension in rethink_module.RETHINK_DIMENSIONS
        ],
    }


class _AdaptiveBatchClient:
    base_url = "https://api.deepseek.com"

    def __init__(self) -> None:
        self.chat = self
        self.completions = self
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def create(self, **kwargs):
        messages = kwargs["messages"]
        reasoning = len(messages) == 2 and messages[-1]["role"] == "user"
        if reasoning:
            context = json.loads(messages[-1]["content"])
            candidates = context["candidates"]
            payload = {"reflections": [_reflection(item) for item in candidates]}
        else:
            payload = json.loads(messages[-2]["content"])
            candidates = payload["reflections"]
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(0.02)
            thinking = kwargs.get("extra_body", {}).get("thinking", {}).get("type")
            call = {
                "stage": "reasoning" if reasoning else "render",
                "batch_size": len(candidates),
                "thinking": thinking,
                "reasoning_effort": kwargs.get("reasoning_effort"),
                "max_tokens": kwargs["max_tokens"],
            }
            with self._lock:
                self.calls.append(call)
            finish_reason = "length" if reasoning and len(candidates) == 8 else "stop"
            content = "" if finish_reason == "length" else json.dumps(payload)
            message = type("Message", (), {"content": content})()
            choice = type(
                "Choice",
                (),
                {"message": message, "finish_reason": finish_reason},
            )()
            return type("Response", (), {"choices": [choice], "usage": None})()
        finally:
            with self._lock:
                self._active -= 1


class _DimensionClient:
    base_url = "https://api.deepseek.com"

    def __init__(self) -> None:
        self.chat = self
        self.completions = self
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def create(self, **kwargs):
        payload = json.loads(kwargs["messages"][-1]["content"])
        dimensions = payload["required_dimensions"]
        variant_id = payload["candidates"][0]["variant_id"]
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(0.01)
            with self._lock:
                self.calls.append(kwargs)
            content = json.dumps(
                {
                    "variant_id": variant_id,
                    "dimension_assessments": [
                        {
                            "dimension": dimension,
                            "evidence_status": "context",
                            "finding": f"Bounded finding for {dimension}.",
                            "implication": "Use only for this sample and round.",
                        }
                        for dimension in dimensions
                    ],
                    "group_advice": "Collect the next discriminating observation.",
                }
            )
            message = type("Message", (), {"content": content})()
            choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
            return type("Response", (), {"choices": [choice], "usage": None})()
        finally:
            with self._lock:
                self._active -= 1


def _context(count: int) -> ReThinkContextInput:
    return ReThinkContextInput(
        run_id="run:adaptive-rethink",
        round_id=1,
        visible_baseline=0.0,
        baseline_receipt={
            "value": 0.0,
            "statistic": "pre_round_visible_median",
            "source": "revealed_observations_before_current_round",
        },
        measurement_contract={
            "assay_id": "assay:test",
            "fitness_scale": "raw_assay",
            "optimization_direction": "higher_is_better",
        },
        final_critic_decision={
            "decision_id": "D01-00",
            "verdict": "APPROVE",
            "summary": "approved",
            "cited_evidence_ids": [],
        },
        candidates=[
            {
                "variant_id": f"V{index:02d}",
                "mutation_notation": f"V39{chr(65 + index)}",
                "wet_value": float(index),
                "dry_validations": [],
                "agent_reason": "bounded reason",
                "evidence_ids": [],
                "intent_arm": "coverage_exploration",
                "allow_hypothesis_mismatch": False,
                "falsification_role": "not_in_primary_criterion",
            }
            for index in range(count)
        ],
    )


def test_rethink_splits_only_failed_batches_and_keeps_reasoning_enabled(monkeypatch) -> None:
    transport_client = _AdaptiveBatchClient()
    monkeypatch.setattr(
        rethink_module,
        "create_openai_client",
        lambda **_kwargs: transport_client,
    )
    monkeypatch.setattr(rethink_module, "report_event", lambda *_args, **_kwargs: None)
    client = NativeReThinkClient(
        model="deepseek-v4-flash",
        provider="deepseek",
        max_tokens=20000,
        reasoning_effort="high",
        thinking="enabled",
        max_transport_retries=0,
        max_truncation_retries=1,
        max_syntax_retries=0,
        max_schema_retries=0,
        max_semantic_retries=0,
        max_unknown_evidence_retries=0,
        retry_backoff_seconds=0.0,
        reasoning_batch_size=8,
        max_parallel_batches=4,
        dimension_parallel=False,
    )

    output = client.reflect_round(context=_context(10))

    assert [item.variant_id for item in output] == [
        f"V{index:02d}" for index in range(10)
    ]
    reasoning_calls = [item for item in transport_client.calls if item["stage"] == "reasoning"]
    render_calls = [item for item in transport_client.calls if item["stage"] == "render"]
    assert sorted(item["batch_size"] for item in reasoning_calls) == [2, 4, 4, 8]
    assert render_calls == []
    assert all(item["thinking"] == "enabled" for item in reasoning_calls)
    assert all(item["reasoning_effort"] == "high" for item in reasoning_calls)
    assert all(item["max_tokens"] == 20000 for item in transport_client.calls)
    assert transport_client.max_active >= 2


def test_rethink_deepseek_defaults_use_bounded_output_and_batch_size(monkeypatch) -> None:
    monkeypatch.setattr(rethink_module, "create_openai_client", lambda **_kwargs: object())
    client = NativeReThinkClient(model="deepseek-chat", provider="deepseek")

    assert client.max_tokens == 20000
    assert client.reasoning_batch_size == 1
    assert client.max_parallel_batches == 8
    assert client.dimension_parallel is True


def test_rethink_parallel_dimension_groups_cover_every_sample(monkeypatch) -> None:
    transport_client = _DimensionClient()
    monkeypatch.setattr(
        rethink_module,
        "create_openai_client",
        lambda **_kwargs: transport_client,
    )
    client = NativeReThinkClient(
        model="deepseek-v4-flash",
        provider="deepseek",
        thinking="enabled",
        reasoning_effort="high",
        max_tokens=20000,
        max_transport_retries=0,
        max_truncation_retries=0,
        max_syntax_retries=0,
        max_schema_retries=0,
        max_semantic_retries=0,
        max_unknown_evidence_retries=0,
        max_parallel_batches=8,
        dimension_parallel=True,
    )

    output = client.reflect_round(context=_context(2))

    assert len(transport_client.calls) == 8
    assert transport_client.max_active >= 2
    assert [item.variant_id for item in output] == ["V00", "V01"]
    assert all(len(item.dimension_assessments) == 8 for item in output)
    assert all(call["max_tokens"] == 20000 for call in transport_client.calls)
