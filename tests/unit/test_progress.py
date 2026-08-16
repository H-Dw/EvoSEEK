from __future__ import annotations

import json
from pathlib import Path

from fitness_agents.agents.remote_llm import complete_json
from fitness_agents.loop import run_campaign
from fitness_agents.utils.artifacts import JsonArtifactWriter
from fitness_agents.utils.progress import bind_progress, reset_progress


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
