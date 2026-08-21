from __future__ import annotations

import pytest

from fitness_agents.agents import rethink as rethink_module
from fitness_agents.agents.output_contracts import ReThinkDimensionGroupOutput
from fitness_agents.agents.rethink import (
    MockReThinkClient,
    NativeReThinkClient,
    _rethink_bridge,
)
from fitness_agents.contracts.agent_io import HypothesisReflectionContextInput


def _context(count: int, *, assessed: bool = True) -> HypothesisReflectionContextInput:
    observations = [
        {
            "variant_id": f"V{index:02d}",
            "mutation_notation": f"V39{chr(65 + index)}",
            "wet_value": float(index),
            "dry_validations": [],
            "evidence_ids": [f"E{index:02d}"],
            "intent_arm": "hypothesis_target",
            "allow_hypothesis_mismatch": False,
            "falsification_role": "target",
        }
        for index in range(count)
    ]
    hypothesis = {
        "hypothesis_id": "H01",
        "statement": "The tested edit pattern improves fitness.",
        "expected_outcome": "Target variants exceed the baseline.",
        "falsification_criterion": "The target mean does not exceed baseline.",
        "evidence_ids": [],
    }
    assessment = {
        "assessment_id": "HA01",
        "hypothesis_id": "H01",
        "falsification_spec_id": "FS01",
        "status": "SUPPORTED",
        "criterion_results": [
            {
                "criterion_id": "C01",
                "signal": "SUPPORT",
                "metric_value": 1.0,
                "observation_ids": [item["variant_id"] for item in observations],
                "qc_status": "ok",
                "detector_name": "mean_delta",
                "detector_version": "v1",
                "reason_code": "target_above_baseline",
            }
        ],
        "observation_ids": [item["variant_id"] for item in observations],
        "decisive_criterion_ids": ["C01"],
        "unresolved_criterion_ids": [],
        "evaluator_version": "v1",
    }
    spec = {
        "spec_id": "FS01",
        "hypothesis_id": "H01",
        "version": "v1",
        "reduction_policy": "primary",
        "criteria": [
            {
                "criterion_id": "C01",
                "detector_name": "mean_delta",
                "metric": "fitness",
                "expected_direction": "higher",
                "target_variant_ids": [item["variant_id"] for item in observations],
                "comparator_variant_ids": [],
                "min_observations": 1,
                "missing_data_policy": "unresolved",
                "primary": True,
            }
        ],
    }
    return HypothesisReflectionContextInput.model_validate(
        {
            "run_id": "run:rethink",
            "round_id": 1,
            "visible_baseline": 0.0,
            "baseline_receipt": {
                "value": 0.0,
                "statistic": "pre_round_visible_median",
                "source": "revealed_observations_before_current_round",
            },
            "measurement_contract": {
                "assay_id": "assay:test",
                "fitness_scale": "raw_assay",
                "optimization_direction": "higher_is_better",
            },
            "approved_hypothesis": hypothesis if assessed else None,
            "hypothesis_assessment": assessment if assessed else None,
            "falsification_spec": spec if assessed else None,
            "final_critic_decision": {
                "decision_id": "D01",
                "verdict": "APPROVE",
                "summary": "approved",
                "cited_evidence_ids": [],
            },
            "round_evidence_digest": {
                "observation_count": count,
                "observations": observations,
                "arm_summaries": [],
                "dry_wet_disagreements": [],
                "criterion_receipts": assessment["criterion_results"] if assessed else [],
                "evidence_ids": [item for row in observations for item in row["evidence_ids"]],
            },
        }
    )


def _group(group_name: str, dimensions: tuple[str, str]) -> ReThinkDimensionGroupOutput:
    return ReThinkDimensionGroupOutput.model_validate(
        {
            "hypothesis_id": "H01",
            "assessment_id": "HA01",
            "group_name": group_name,
            "dimension_assessments": [
                {
                    "dimension": dimension,
                    "evidence_status": "measured",
                    "relation_to_hypothesis": "positive",
                    "finding_code": "supported",
                    "finding": "The round-level receipt supports the bounded claim.",
                    "implication": "Retain the claim only in the observed scope.",
                }
                for dimension in dimensions
            ],
            "retained_claims": ["Retain the tested claim in scope."],
            "recommended_actions": ["retain_uncertainty_aware_exploration"],
            "supporting_observation_ids": ["V00"],
            "supporting_evidence_ids": ["E00"],
            "group_advice": "Keep the conclusion hypothesis-level and bounded.",
        }
    )


def _client(monkeypatch, **kwargs) -> NativeReThinkClient:
    monkeypatch.setattr(rethink_module, "create_openai_client", lambda **_kw: object())
    return NativeReThinkClient(
        model="unit-test-model",
        provider="openai_compatible",
        thinking="disabled",
        max_transport_retries=0,
        max_truncation_retries=0,
        max_syntax_retries=0,
        max_schema_retries=0,
        max_semantic_retries=0,
        max_unknown_evidence_retries=0,
        retry_backoff_seconds=0.0,
        parallel_dimension_groups=False,
        **kwargs,
    )


@pytest.mark.parametrize("sample_count", [1, 2, 25])
def test_rethink_uses_exactly_four_calls_independent_of_sample_count(
    monkeypatch, sample_count: int
) -> None:
    client = _client(monkeypatch)
    calls: list[str] = []

    def fake_group(*, context, group_name, dimensions):
        del context
        calls.append(group_name)
        return _group(group_name, dimensions)

    monkeypatch.setattr(client, "_reflect_dimension_group", fake_group)
    reflection = client.reflect_hypothesis(context=_context(sample_count))

    assert reflection is not None
    assert reflection.hypothesis_id == "H01"
    assert len(calls) == 4
    assert set(calls) == set(rethink_module.RETHINK_DIMENSION_GROUPS)
    assert len(reflection.dimension_assessments) == 8


def test_one_dimension_group_failure_degrades_only_that_group(monkeypatch) -> None:
    client = _client(monkeypatch)

    def fake_group(*, context, group_name, dimensions):
        del context
        if group_name == "sequence_and_physchem":
            raise RuntimeError("synthetic failure")
        return _group(group_name, dimensions)

    monkeypatch.setattr(client, "_reflect_dimension_group", fake_group)
    reflection = client.reflect_hypothesis(context=_context(3))

    assert reflection is not None
    degraded = [
        item
        for item in reflection.dimension_assessments
        if item["quality_status"] == "deterministic_fallback"
    ]
    assert {item["dimension"] for item in degraded} == {
        "sequence_interaction_context",
        "physicochemical_context",
    }
    assert reflection.quality_status == "deterministic_fallback"


def test_rethink_skips_llm_when_no_assessed_hypothesis(monkeypatch) -> None:
    client = _client(monkeypatch)
    monkeypatch.setattr(
        client,
        "_reflect_dimension_group",
        lambda **_kwargs: pytest.fail("LLM dimension call should not run"),
    )

    assert client.reflect_hypothesis(context=_context(2, assessed=False)) is None


def test_mock_rethink_returns_one_hypothesis_reflection() -> None:
    reflection = MockReThinkClient().reflect_hypothesis(context=_context(20))

    assert reflection is not None
    assert reflection.assessment_id == "HA01"
    assert reflection.advisory_only is True
    assert reflection.selection_eligible is False


def test_rethink_id_bridge_keeps_hypothesis_scope_consistent() -> None:
    context = _context(4)
    bridge = _rethink_bridge(context, scope_id="test-scope")

    aliased = HypothesisReflectionContextInput.model_validate(
        bridge.encode_projection(context.model_dump(mode="python"))
    )

    assert aliased.approved_hypothesis is not None
    assert aliased.hypothesis_assessment is not None
    assert aliased.falsification_spec is not None
    assert (
        aliased.approved_hypothesis.hypothesis_id
        == aliased.hypothesis_assessment.hypothesis_id
        == aliased.falsification_spec.hypothesis_id
    )
    assert (
        aliased.hypothesis_assessment.falsification_spec_id
        == aliased.falsification_spec.spec_id
    )
