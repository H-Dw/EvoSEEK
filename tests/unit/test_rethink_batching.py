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
        "verdict": "mixed",
        "summary": "Wet and dry evidence remain mixed.",
        "positive_findings": ["A bounded positive signal was observed."],
        "negative_findings": ["Uncertainty remains material."],
        "revised_reason": "Retain the rationale only as round-specific evidence.",
        "next_round_advice": "Test a matched alternative in the next round.",
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


def _context(count: int) -> ReThinkContextInput:
    return ReThinkContextInput(
        run_id="run:adaptive-rethink",
        round_id=1,
        visible_baseline=0.0,
        candidates=[
            {
                "variant_id": f"V{index:02d}",
                "mutation_notation": f"V39{chr(65 + index)}",
                "wet_value": float(index),
                "dry_validations": [],
                "agent_reason": "bounded reason",
                "evidence_ids": [],
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
