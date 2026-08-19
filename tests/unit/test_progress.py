from __future__ import annotations

import json
from pathlib import Path

import pytest

from fitness_agents.agents.remote_llm import RemoteLLMCompletionError, complete_json
from fitness_agents.contracts.batch_review import control_feasibility_receipt
from fitness_agents.loop import run_campaign
from fitness_agents.utils.artifacts import JsonArtifactWriter
from fitness_agents.utils.progress import bind_progress, reset_progress


def test_artifact_writer_serializes_typed_pydantic_receipts(tmp_path: Path) -> None:
    writer = JsonArtifactWriter(tmp_path, "typed-receipt-run")
    receipt = control_feasibility_receipt(
        requested_controls=2,
        available_control_ids=("control:1", "control:2"),
        selected_control_ids=("control:1", "control:2"),
    )
    path = writer.write_json("receipt.json", receipt)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["reason"] == "FEASIBLE"
    assert payload["selected_controls"] == 2


def test_status_json_tracks_heartbeat_without_trace_noise(tmp_path: Path) -> None:
    writer = JsonArtifactWriter(tmp_path, "progress-run")
    writer.heartbeat("encoding batch", log=False, phase="predicting", round_id=2, completed=3, total=10)
    writer.report(
        "predict_started",
        message="predicting candidates",
        persist=True,
        phase="predicting",
        round_id=2,
        n_candidates=88,
    )
    writer.event("batch_selected", {"records": ["audit-only"]})

    status = json.loads((tmp_path / "progress-run" / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "predicting"
    assert status["round_id"] == 2
    assert status["message"] == "predicting candidates"
    assert status["event_type"] == "predict_started"
    assert "n_candidates" in status["detail"]

    events = [
        json.loads(line)["event_type"]
        for line in (tmp_path / "progress-run" / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert events == ["predict_started", "batch_selected"]
    trace = (tmp_path / "progress-run" / "trace.jsonl").read_text(encoding="utf-8")
    assert "audit-only" in trace


def test_campaign_emits_started_completed_progress_events(config_factory) -> None:
    config = config_factory(
        mode="knowledge_agent",
        acquisition="greedy",
        knowledge_enabled=True,
        candidate_limit=40,
        rounds=1,
        budget_per_round=3,
        run_label="progress",
    )
    summary = run_campaign(config)
    run_dir = Path(summary["run_dir"])
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "finalized"
    assert status["message"] == "campaign finalized"

    events = [
        json.loads(line)["event_type"]
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    required = (
        "campaign_started",
        "round_started",
        "model_fit_started",
        "model_fit_completed",
        "predict_started",
        "predict_completed",
        "evidence_started",
        "hypothesis_generation_started",
        "review_attempt_started",
        "critique_started",
        "round_completed",
        "campaign_finalized",
    )
    missing = [name for name in required if name not in events]
    assert missing == []
    started = events.index("review_attempt_started")
    drafted = events.index("batch_drafted")
    critique_started = events.index("critique_started")
    critique_completed = events.index("critique_completed")
    assert started < drafted < critique_started < critique_completed


class _FakeClient:
    base_url = "https://example.invalid/v1"

    def __init__(self) -> None:
        self.chat = self

    def create(self, **kwargs):
        del kwargs
        message = type("Message", (), {"content": '{"ok": true}'})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "completion_tokens_details": None,
            },
        )()
        return type("Response", (), {"choices": [choice], "usage": usage})()

    @property
    def completions(self):
        return self


class _CaptureProgress:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str]] = []

    def heartbeat(self, message: str, *, log: bool = True, **payload) -> None:
        self.calls.append(("heartbeat", message))

    def report(
        self,
        event_type: str | None,
        *,
        message: str,
        persist: bool = True,
        **payload,
    ) -> None:
        del persist, payload
        self.calls.append((event_type, message))


def test_complete_json_reports_request_lifecycle() -> None:
    capture = _CaptureProgress()
    token = bind_progress(capture)
    try:
        payload = complete_json(
            client=_FakeClient(),
            model="unit-test-model",
            messages=[{"role": "user", "content": "return json"}],
        )
    finally:
        reset_progress(token)

    assert payload == {"ok": True}
    event_types = [item[0] for item in capture.calls]
    assert "llm_request_started" in event_types
    assert "llm_request_completed" in event_types


def test_prompt_budget_artifact_contains_only_size_metadata(tmp_path: Path) -> None:
    writer = JsonArtifactWriter(tmp_path, "prompt-budget-run")
    writer.write_status(message="round active", phase="llm_hypothesis", round_id=1)
    token = bind_progress(writer)
    try:
        payload = complete_json(
            client=_FakeClient(),
            model="unit-test-model",
            messages=[
                {"role": "system", "content": "system contract"},
                {"role": "user", "content": '{"context":{"secret":"do-not-persist"}}'},
            ],
            max_input_chars=1000,
            trace_context={
                "role": "subscientist:physchem",
                "profile": "physchem_v1",
                "round_id": 1,
            },
        )
    finally:
        reset_progress(token)

    assert payload == {"ok": True}
    path = (
        tmp_path
        / "prompt-budget-run"
        / "round_01"
        / "llm"
        / "subscientist_physchem"
        / "prompt_budget.json"
    )
    records = json.loads(path.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert set(records[0]) == {
        "role",
        "profile",
        "system_chars",
        "user_chars",
        "assistant_chars",
        "input_chars",
        "field_chars",
        "max_input_chars",
        "remaining_chars",
        "utilization_ratio",
        "budget_band",
        "request_started",
    }
    assert records[0]["request_started"] is True
    assert records[0]["budget_band"] == "normal"
    assert records[0]["input_chars"] == (
        records[0]["system_chars"] + records[0]["user_chars"]
    )
    assert records[0]["field_chars"]["user.context"] > 0
    assert "do-not-persist" not in path.read_text(encoding="utf-8")


def test_prompt_budget_artifact_marks_preflight_rejection_without_request(tmp_path: Path) -> None:
    writer = JsonArtifactWriter(tmp_path, "prompt-budget-rejected")
    writer.write_status(message="round active", phase="llm_hypothesis", round_id=2)
    token = bind_progress(writer)
    try:
        with pytest.raises(RemoteLLMCompletionError, match="PROMPT_BUDGET_EXCEEDED"):
            complete_json(
                client=_FakeClient(),
                model="unit-test-model",
                messages=[{"role": "user", "content": '{"evidence":"' + "x" * 100 + '"}'}],
                max_input_chars=20,
                trace_context={"role": "scientist", "profile": "scientific_v1", "round_id": 2},
            )
    finally:
        reset_progress(token)

    path = (
        tmp_path
        / "prompt-budget-rejected"
        / "round_02"
        / "llm"
        / "scientist"
        / "prompt_budget.json"
    )
    records = json.loads(path.read_text(encoding="utf-8"))
    assert records[0]["request_started"] is False
    assert records[0]["user_chars"] > records[0]["max_input_chars"]
    assert records[0]["budget_band"] == "exceeded"


def test_prompt_budget_high_water_emits_size_only_warning(tmp_path: Path) -> None:
    writer = JsonArtifactWriter(tmp_path, "prompt-budget-warning")
    writer.write_status(message="round active", phase="llm_hypothesis", round_id=3)
    token = bind_progress(writer)
    try:
        complete_json(
            client=_FakeClient(),
            model="unit-test-model",
            messages=[{"role": "user", "content": '{"context":"' + "x" * 820 + '"}'}],
            max_input_chars=1000,
            trace_context={"role": "scientist", "round_id": 3},
        )
    finally:
        reset_progress(token)

    events = [
        json.loads(line)
        for line in (tmp_path / "prompt-budget-warning" / "trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    warning = next(
        item for item in events if item["event_type"] == "llm_prompt_budget_high_water"
    )
    assert warning["payload"]["budget_band"] == "warning"
    assert "x" * 20 not in json.dumps(warning)
