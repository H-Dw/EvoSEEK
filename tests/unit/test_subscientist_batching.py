from __future__ import annotations

import json
import threading
import time

import pytest

from fitness_agents.agents import subscientist as subscientist_module
from fitness_agents.agents.adaptive_batch import (
    AdaptiveBatchExecutionError,
    adaptive_batch_submit,
)
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
    PhyschemInterpretationOutput,
)


def _context(count: int) -> ChannelEvidenceInput:
    residues = "ACDEFGHIKLMNPQRSTVWY"
    return ChannelEvidenceInput(
        run_id="run:subscientist-batch",
        round_id=1,
        channel="physchem",
        task="summarize channel-local feature evidence",
        mutable_positions=(39, 40, 41, 54),
        wild_type_sites="VDGV",
        sample_map={f"sample:{index}": f"V39{residues[index % 20]}" for index in range(count)},
        visible_observations=tuple(
            {
                "sample_id": f"sample:{index}",
                "mutation_notation": f"V39{residues[index % 20]}",
                "residues_by_position": {"39": residues[index % 20]},
                "evidence_ids": (f"ev:pc:{index}",),
                "feature_values": {
                    f"ev:pc:{index}": {"charge_delta": float(index)}
                },
                "descriptor_facts": (
                    {
                        "fact_id": f"D{index + 1:03d}",
                        "evidence_id": f"ev:pc:{index}",
                        "sample_id": f"sample:{index}",
                        "position": 39,
                        "from_residue": "V",
                        "to_residue": residues[index % 20],
                        "descriptor": "charge_delta",
                        "delta": float(index),
                    },
                ),
            }
            for index in range(count)
        ),
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
            return PhyschemInterpretationOutput(
                analysis_summary=f"Analyzed {len(sample_ids)} sample-local descriptor cards.",
                interpretations=["The visible charge deltas span bounded directions."],
                counterevidence=[],
                uncertainty="Descriptor deltas do not establish assay performance.",
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
    assert len(result.analysis.findings) == 8
    assert len({item.finding_id for item in result.analysis.findings}) == 8
    assert {
        item.analysis.findings[0].finding_id for item in result.batches
    } == {"F01"}
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
            "mutation_notation": f"V39{chr(65 + index)}",
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


def test_subscientist_projection_uses_one_runtime_sample_id() -> None:
    context = ChannelEvidenceInput.model_validate(
        {
            **_context(1).model_dump(mode="json"),
            "visible_observations": [
                {
                    "sample_id": "sample:0",
                    "mutation_notation": "V39A",
                    "residues_by_position": {"39": "A"},
                    "evidence_ids": ("ev:pc:0",),
                    "feature_values": {"ev:pc:0": {"charge_delta": 0}},
                }
            ],
        }
    )

    sample_ids = subscientist_module._context_sample_ids(context)
    projected = subscientist_module._batch_context(context, sample_ids=sample_ids)

    assert sample_ids == ("sample:0",)
    assert projected.visible_observations[0].sample_id == "sample:0"
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


def test_physchem_materialization_keeps_multi_mutation_facts_separate() -> None:
    raw = _context(1).model_dump(mode="python")
    raw["visible_observations"][0]["mutation_notation"] = "V39A;D40Y"
    raw["visible_observations"][0]["descriptor_facts"] = (
        {
            "fact_id": "D001",
            "evidence_id": "ev:pc:0",
            "sample_id": "sample:0",
            "position": 39,
            "from_residue": "V",
            "to_residue": "A",
            "descriptor": "charge_delta",
            "delta": 1.0,
        },
        {
            "fact_id": "D002",
            "evidence_id": "ev:pc:0",
            "sample_id": "sample:0",
            "position": 40,
            "from_residue": "D",
            "to_residue": "Y",
            "descriptor": "mass_delta",
            "delta": 2.0,
        },
    )
    context = ChannelEvidenceInput.model_validate(raw)
    output = subscientist_module._materialize_physchem_analysis(
        context=context,
        explanation=PhyschemInterpretationOutput(
            analysis_summary="Two mutation-scoped descriptor cards are visible.",
            interpretations=[],
            counterevidence=[],
            uncertainty="Descriptor changes do not establish assay performance.",
        ),
        batch_id="b000",
    )

    observations = [item for item in output.findings if item.kind == "OBSERVATION"]
    assert len(observations) == 2
    assert observations[0].fact_ids == ["D001"]
    assert "V39A" in observations[0].statement
    assert observations[1].fact_ids == ["D002"]
    assert "D40Y" in observations[1].statement


def test_adaptive_batch_failure_preserves_completed_sibling_results() -> None:
    def submit(work):
        if work.item_ids == ("failed",):
            time.sleep(0.01)
            raise ValueError("synthetic local failure")
        time.sleep(0.02)
        return work.item_ids[0]

    with pytest.raises(AdaptiveBatchExecutionError) as captured:
        adaptive_batch_submit(
            ("failed", "succeeded"),
            item_id=str,
            submit_batch=submit,
            initial_batch_size=1,
            max_parallel_batches=2,
            should_split_failure=lambda _error: False,
            role="test",
            event_reporter=lambda *_args, **_kwargs: None,
            preserve_completed_on_failure=True,
        )

    assert [item.output for item in captured.value.completed] == ["succeeded"]
