from __future__ import annotations

import json
import threading
import time

import pytest

from fitness_agents.agents import subscientist as subscientist_module
from fitness_agents.agents.output_guards import SemanticOutputValidationError
from fitness_agents.agents.remote_llm import RemoteLLMCompletionError
from fitness_agents.agents.subscientist import (
    RemoteSubScientist,
    validate_channel_hypothesis,
)
from fitness_agents.contracts.hypothesis_pipeline import (
    BatchedChannelAnalysisResult,
    ChannelAnalysisOutput,
    ChannelEvidenceInput,
)


def _context(count: int) -> ChannelEvidenceInput:
    return ChannelEvidenceInput(
        run_id="run:subscientist-batch",
        round_id=1,
        channel="physchem",
        task="summarize channel-local feature evidence",
        mutable_positions=(39, 40, 41, 54),
        wild_type_sites="VDGV",
        evidence=tuple(
            {
                "evidence_id": f"ev:pc:{index}",
                "variant_id": f"sample:{index}",
                "channel": "physchem",
                "statement": f"Sample {index} has a bounded descriptor delta.",
            }
            for index in range(count)
        ),
    )


def _client() -> RemoteSubScientist:
    client = object.__new__(RemoteSubScientist)
    client.profile_name = "physchem_v1"
    client.profile = "Analyze only physicochemical evidence."
    client.profile_sha256 = "profile-sha"
    client.model = "deepseek-v4-flash"
    client.temperature = 0.0
    client.max_tokens = 20000
    client.reasoning_effort = None
    client.thinking = "disabled"
    client.max_transport_retries = 0
    client.max_truncation_retries = 0
    client.max_syntax_retries = 0
    client.max_schema_retries = 1
    client.max_semantic_retries = 1
    client.max_unknown_evidence_retries = 1
    client.retry_backoff_seconds = 0.0
    client.allow_unknown_evidence_stripping = False
    client.max_input_chars = 160000
    client.sample_batch_size = 8
    client.max_parallel_batches = 4
    client.client = object()
    client.transport = object()
    return client


def test_subscientist_uses_shared_batch_submit_and_persists_typed_batch_outputs(
    monkeypatch,
) -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0
    call_sizes: list[int] = []

    def complete(**kwargs):
        nonlocal active, max_active
        payload = json.loads(kwargs["messages"][-1]["content"])
        batch_context = payload["immutable_channel_context"]
        evidence = batch_context["evidence"]
        sample_ids = [item["variant_id"] for item in evidence]
        with lock:
            active += 1
            max_active = max(max_active, active)
            call_sizes.append(len(sample_ids))
        try:
            time.sleep(0.02)
            if len(sample_ids) == 8:
                raise RemoteLLMCompletionError(
                    "OUTPUT_TRUNCATED",
                    failure_category="output",
                    input_chars=1000,
                    request_started=True,
                    detail="synthetic length boundary",
                )
            evidence_ids = [item["evidence_id"] for item in evidence]
            return ChannelAnalysisOutput(
                analysis_id=f"analysis:{sample_ids[0]}",
                channel="physchem",
                analysis_summary=f"Analyzed {len(sample_ids)} sample-local descriptor cards.",
                findings=[
                    {
                        "finding_id": "finding:batch-local:1",
                        "kind": "OBSERVATION",
                        "statement": "The supplied samples have bounded descriptor deltas.",
                        "evidence_ids": evidence_ids[:8],
                        "confidence": "medium",
                    }
                ],
                candidate_hypotheses=[],
                evidence_ids=evidence_ids[:12],
                counterevidence=[],
                uncertainty="Descriptor deltas do not establish fitness.",
            )
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(subscientist_module, "complete_structured", complete)
    monkeypatch.setattr(subscientist_module, "report_event", lambda *_args, **_kwargs: None)

    result = _client().propose(context=_context(10))

    assert isinstance(result, BatchedChannelAnalysisResult)
    assert sorted(call_sizes) == [2, 4, 4, 8]
    assert [len(item.sample_ids) for item in result.batches] == [4, 4, 2]
    assert {
        sample_id for item in result.batches for sample_id in item.sample_ids
    } == {f"sample:{index}" for index in range(10)}
    assert result.analysis.channel == "physchem"
    assert len(result.analysis.findings) == 3
    assert len({item.finding_id for item in result.analysis.findings}) == 3
    assert {
        item.analysis.findings[0].finding_id for item in result.batches
    } == {"finding:batch-local:1"}
    assert "covering 10 samples" in result.analysis.analysis_summary
    assert max_active >= 2


def test_subscientist_sample_projection_bounds_a_64_sample_prompt() -> None:
    context = _context(64)
    sample_ids = subscientist_module._context_sample_ids(context)
    batches = [
        subscientist_module._batch_context(
            context,
            sample_ids=sample_ids[offset : offset + 8],
        )
        for offset in range(0, len(sample_ids), 8)
    ]

    full_chars = len(context.model_dump_json())
    batch_chars = [len(item.model_dump_json()) for item in batches]
    assert len(batches) == 8
    assert all(len(item.evidence) == 8 for item in batches)
    assert max(batch_chars) < full_chars * 0.25


def test_subscientist_batches_only_samples_with_role_visible_evidence() -> None:
    visible_observations = tuple(
        {
            "sample_id": f"sample:{index}",
            "variant_id": f"sample:{index}",
            "mutation_notation": f"V39{chr(65 + index)}",
            "sequence_sha256": str(index) * 64,
            "residues_by_position": {"39": chr(65 + index)},
            "evidence_ids": (f"ev:pc:{index}",) if index < 2 else (),
            "feature_values": (
                {f"ev:pc:{index}": {"charge_delta": index}} if index < 2 else {}
            ),
        }
        for index in range(10)
    )
    context = ChannelEvidenceInput.model_validate(
        {
            **_context(2).model_dump(mode="json"),
            "visible_observations": visible_observations,
        }
    )

    sample_ids = subscientist_module._context_sample_ids(context)
    projected = subscientist_module._batch_context(context, sample_ids=sample_ids)

    assert sample_ids == ("sample:0", "sample:1")
    assert [item.sample_id for item in projected.visible_observations] == [
        "sample:0",
        "sample:1",
    ]
    assert len(projected.evidence) == 2


def test_subscientist_projection_expands_sample_and_variant_aliases() -> None:
    context = ChannelEvidenceInput.model_validate(
        {
            **_context(1).model_dump(mode="json"),
            "visible_observations": [
                {
                    "sample_id": "row:0",
                    "variant_id": "sample:0",
                    "mutation_notation": "V39A",
                    "sequence_sha256": "a" * 64,
                    "residues_by_position": {"39": "A"},
                    "evidence_ids": ("ev:pc:0",),
                    "feature_values": {"ev:pc:0": {"charge_delta": 0}},
                }
            ],
        }
    )

    sample_ids = subscientist_module._context_sample_ids(context)
    projected = subscientist_module._batch_context(context, sample_ids=sample_ids)

    assert sample_ids == ("row:0",)
    assert projected.visible_observations[0].variant_id == "sample:0"
    assert projected.evidence[0]["evidence_id"] == "ev:pc:0"


def test_subscientist_batch_projection_honors_pack_metadata_variant_id() -> None:
    context = _context(2).model_copy(
        update={
            "kg_packs": (
                {
                    "operator": "query_physchem_delta",
                    "metadata": {"variant_id": "sample:0"},
                    "evidence": [
                        {
                            "evidence_id": "ev:pack:0",
                            "channel": "physchem",
                            "statement": "Sample 0 descriptor delta.",
                        }
                    ],
                },
                {
                    "operator": "query_physchem_delta",
                    "metadata": {"variant_id": "sample:1"},
                    "evidence": [
                        {
                            "evidence_id": "ev:pack:1",
                            "channel": "physchem",
                            "statement": "Sample 1 descriptor delta.",
                        }
                    ],
                },
            )
        }
    )

    projected = subscientist_module._batch_context(
        context,
        sample_ids=("sample:0",),
    )

    assert len(projected.kg_packs) == 1
    assert projected.kg_packs[0]["metadata"]["variant_id"] == "sample:0"
    assert projected.visible_evidence_ids == frozenset({"ev:pc:0", "ev:pack:0"})


def test_non_limitation_finding_requires_a_visible_evidence_link() -> None:
    payload = ChannelAnalysisOutput(
        analysis_id="analysis:uncited",
        channel="physchem",
        analysis_summary="A descriptor observation was attempted.",
        findings=[
            {
                "finding_id": "finding:uncited",
                "kind": "OBSERVATION",
                "statement": "G41E introduces a nominal negative charge.",
                "evidence_ids": [],
                "confidence": "medium",
            }
        ],
        candidate_hypotheses=[],
        evidence_ids=[],
        counterevidence=[],
        uncertainty="No linked descriptor card is visible for this statement.",
    )

    with pytest.raises(SemanticOutputValidationError) as captured:
        validate_channel_hypothesis(payload.model_dump(mode="json"), context=_context(1))

    assert captured.value.paths == (
        "findings.0.kind",
        "findings.0.evidence_ids",
    )
