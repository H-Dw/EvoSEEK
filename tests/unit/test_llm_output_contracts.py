from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from fitness_agents.agents.llm import (
    HYPOTHESIS_SCHEMA,
    MockScientistLLMClient,
    OpenAICompatibleLLMClient,
    build_scientist_hypothesis_messages,
    load_scientist_profile,
)
from fitness_agents.agents.output_contracts import HypothesisOutput, validate_hypothesis_payload
from fitness_agents.agents.output_guards import UnknownEvidenceIdsError
from fitness_agents.agents.rethink import NativeReThinkClient
from fitness_agents.agents.transports import OpenAICompatibleChatTransport
from fitness_agents.contracts.agent_io import ReThinkContextInput
from fitness_agents.contracts.schemas import Evidence


class _SequenceClient:
    base_url = "https://example.invalid/v1"

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.calls = 0
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        del kwargs
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        message = type("Message", (), {"content": json.dumps(payload)})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type("Response", (), {"choices": [choice], "usage": None})()


def _valid_payload() -> dict:
    return {
        "hypothesis_id": "hyp:run:r1",
        "statement": "Visible evidence supports a bounded four-site hypothesis.",
        "preferred_residues": {
            "39": ["W"],
            "40": ["D"],
            "41": ["G"],
            "54": ["V"],
        },
        "evidence_ids": [],
        "expected_outcome": "Enrichment relative to random selection.",
        "falsification_criterion": "Revise if the wet batch median does not improve.",
        "parent_hypothesis_id": None,
    }


def _client(remote: _SequenceClient) -> OpenAICompatibleLLMClient:
    client = OpenAICompatibleLLMClient.__new__(OpenAICompatibleLLMClient)
    client.model = "unit-test-model"
    client.temperature = 0.0
    client.max_tokens = 1024
    client.reasoning_effort = None
    client.thinking = None
    client.profile = "Return the complete hypothesis contract."
    client.client = remote
    return client


def _context() -> dict:
    return {
        "run_id": "run",
        "mode": "knowledge_agent",
        "round_id": 1,
        "expected_hypothesis_id": "hyp:run:r1",
        "task": "maximize visible fitness",
        "protein_id": "GB1",
        "objective": "maximize",
        "mutable_positions": [39, 40, 41, 54],
        "wild_type_sites": "VDGV",
        "protein_context_id": "ctx:test",
        "visible_observations": [],
        "previous_hypothesis_id": None,
        "previous_hypothesis_assessment": None,
    }


def test_missing_hypothesis_id_retries_inside_json_boundary() -> None:
    missing_id = _valid_payload()
    del missing_id["hypothesis_id"]
    remote = _SequenceClient([missing_id, _valid_payload()])

    hypothesis = _client(remote).generate_hypothesis(
        sanitized_context=_context(),
        evidence=[],
        output_schema=HYPOTHESIS_SCHEMA,
    )

    assert remote.calls == 2
    assert hypothesis.hypothesis_id == "hyp:run:r1"


def test_exhausted_missing_key_is_validation_error_not_key_error() -> None:
    missing_id = _valid_payload()
    del missing_id["hypothesis_id"]
    remote = _SequenceClient([missing_id])

    with pytest.raises(RuntimeError) as captured:
        _client(remote).generate_hypothesis(
            sanitized_context=_context(),
            evidence=[],
            output_schema=HYPOTHESIS_SCHEMA,
        )

    # Formal runtime permits one output-repair request (two total calls).
    assert remote.calls == 2
    assert isinstance(captured.value.__cause__, ValidationError)


def test_unknown_evidence_ids_are_stripped_after_retries() -> None:
    bad = _valid_payload()
    bad["evidence_ids"] = ["ev:missing"]
    remote = _SequenceClient([bad])
    evidence = Evidence(
        evidence_id="ev:1",
        variant_id="context:protein",
        channel="local_rag",
        statement="Visible evidence.",
        score=0.0,
        source_id="localdoc:1",
        confidence=0.8,
        round_id=1,
    )

    hypothesis = _client(remote).generate_hypothesis(
        sanitized_context=_context(),
        evidence=[evidence],
        output_schema=HYPOTHESIS_SCHEMA,
    )

    assert remote.calls == 2
    assert hypothesis.evidence_ids == ()


def test_campaign_owned_ids_are_coerced_without_retry() -> None:
    payload = _valid_payload()
    payload["hypothesis_id"] = "hyp:invented-by-model"
    payload["parent_hypothesis_id"] = None
    context = {**_context(), "previous_hypothesis_id": "hyp:run:r0"}
    remote = _SequenceClient([payload])

    hypothesis = _client(remote).generate_hypothesis(
        sanitized_context=context,
        evidence=[],
        output_schema=HYPOTHESIS_SCHEMA,
    )

    assert remote.calls == 1
    assert hypothesis.hypothesis_id == "hyp:run:r1"
    assert hypothesis.parent_hypothesis_id == "hyp:run:r0"


def test_native_scientist_revision_block_sets_parent_and_attempt_id() -> None:
    payload = _valid_payload()
    payload["hypothesis_id"] = "hyp:wrong"
    payload["parent_hypothesis_id"] = None
    payload["statement"] = "Revised residue map after critic asked for a new hypothesis."
    context = {
        **_context(),
        "expected_hypothesis_id": "hyp:run:r1:a1",
        "previous_hypothesis_id": "hyp:run:r0",
        "critic_revision": {
            "verdict": "REVISE",
            "summary": "Do not restated the rejected residue map.",
            "rejected_hypothesis_id": "hyp:run:r1",
            "rejected_preferred_residues": {
                "39": ["W"],
                "40": ["D"],
                "41": ["G"],
                "54": ["V"],
            },
        },
    }
    remote = _SequenceClient([payload])

    hypothesis = _client(remote).generate_hypothesis(
        sanitized_context=context,
        evidence=[],
        output_schema=HYPOTHESIS_SCHEMA,
    )

    assert hypothesis.hypothesis_id == "hyp:run:r1:a1"
    assert hypothesis.parent_hypothesis_id == "hyp:run:r1"


def test_hypothesis_output_schema_accepts_task_scoped_dynamic_site_keys() -> None:
    schema = HypothesisOutput.model_json_schema()
    site_schema = schema["properties"]["preferred_residues"]

    assert site_schema["type"] == "object"
    assert site_schema["additionalProperties"]["type"] == "array"


def test_scientist_profile_defines_output_and_authority_boundaries() -> None:
    profile = load_scientist_profile("scientific_v1")

    assert "expected_hypothesis_id" in profile
    assert "oracle" in profile
    assert "final-test" in profile
    assert "batch submission" in profile
    assert "sha256:" in profile
    assert "critic_revision" in profile
    assert "400" in profile
    assert "visible_observations" in profile
    assert "## 3. Activation-state routing" in profile
    assert "## 4. Directed-evolution reasoning hierarchy" in profile
    assert "executed_kg_tools" in profile
    assert "open_design" in profile
    assert "active_learning" in profile
    for forbidden_prior in ("AAIndex", "Neff", "SASA", "salt-bridge", "hydropathy"):
        assert forbidden_prior not in profile
    assert "physicochemical context" in profile
    assert "evolutionary context" in profile
    assert "structural context" in profile
    assert "JSON object" in profile


def test_variant_hash_evidence_ids_are_dropped_unknown_evidence_still_fails() -> None:
    payload = _valid_payload()
    payload["evidence_ids"] = [
        "ev:1",
        "sha256:06f55338c6fc1a65a6ca3d486e6641f52abfaabb0c3353743f1afc323443f61b",
    ]
    cleaned = validate_hypothesis_payload(
        payload,
        allowed_evidence_ids=frozenset({"ev:1"}),
        expected_positions=(39, 40, 41, 54),
    )
    assert cleaned["evidence_ids"] == ["ev:1"]

    empty = validate_hypothesis_payload(
        {**payload, "evidence_ids": ["sha256:deadbeef"]},
        allowed_evidence_ids=frozenset(),
        expected_positions=(39, 40, 41, 54),
    )
    assert empty["evidence_ids"] == []

    with pytest.raises(UnknownEvidenceIdsError, match="not visible to the role"):
        validate_hypothesis_payload(
            {**payload, "evidence_ids": ["ev:missing"]},
            allowed_evidence_ids=frozenset({"ev:1"}),
            expected_positions=(39, 40, 41, 54),
        )

    stripped = validate_hypothesis_payload(
        {**payload, "evidence_ids": ["ev:missing", "ev:1"]},
        allowed_evidence_ids=frozenset({"ev:1"}),
        expected_positions=(39, 40, 41, 54),
        on_unknown_evidence="strip",
    )
    assert stripped["evidence_ids"] == ["ev:1"]


def test_scientist_prompt_keeps_decision_provenance_but_drops_backend_bulk() -> None:
    context = {
        **_context(),
        "kg_interaction": {
            "plan_id": "p1",
            "packs": [
                {
                    "query_id": "q1",
                    "operator": "query_local_knowledge",
                    "as_of_round": 1,
                    "facts": [],
                    "evidence": [
                        {
                            "evidence_id": "ev:1",
                            "statement": "Epistasis is background dependent.",
                            "provenance": {
                                "artifact_uri": "claim.md",
                                "embedding_fingerprint": {"large": "x" * 1000},
                            },
                        }
                    ],
                    "provenance": [],
                }
            ],
        },
    }
    evidence = Evidence(
        evidence_id="ev:1",
        variant_id="context:protein",
        channel="local_rag",
        statement="Epistasis is background dependent.",
        score=0.0,
        source_id="localdoc:1",
        confidence=0.8,
        round_id=1,
        provenance={
            "artifact_uri": "claim.md",
            "index_manifest_hash": "manifest",
            "embedding_fingerprint": {"large": "x" * 1000},
        },
    )

    messages = build_scientist_hypothesis_messages(
        profile="scientist profile",
        sanitized_context=context,
        evidence=[evidence],
        output_schema=HYPOTHESIS_SCHEMA,
    )

    payload = json.loads(messages[1]["content"])
    assert payload["context"]["activation_state"]["design_space"] == "closed_pool"
    assert payload["context"]["activation_state"]["executed_kg_tools"] == []
    assert payload["evidence"][0]["evidence_id"] == "ev:1"
    assert payload["evidence"][0]["provenance"] == {
        "artifact_uri": "claim.md",
        "index_manifest_hash": "manifest",
    }
    assert "embedding_fingerprint" not in messages[1]["content"]


def test_scientist_prompt_keeps_bounded_feature_tool_evidence() -> None:
    context = {
        **_context(),
        "kg_interaction": {
            "plan_id": "p-features",
            "packs": [
                {
                    "query_id": "q-feature",
                    "operator": "query_feature_bundle",
                    "as_of_round": 1,
                    "facts": [],
                    "evidence": [
                        {
                            "evidence_id": "ev:structure",
                            "variant_id": "v1",
                            "channel": "structure",
                            "statement": "Static environment only.",
                            "raw_features": {
                                "sites": {
                                    "39": {
                                        "contact_count": 9,
                                        "relative_sasa": 0.2,
                                    }
                                },
                                "static_context_flag_count": 1,
                                "resource_id": "rcsb:1PGB",
                                "backend_coordinate_dump": "x" * 1000,
                            },
                            "provenance": {
                                "provider": "StaticStructureProvider",
                                "resource_sha256": "abc",
                                "private_cache_state": "x" * 1000,
                            },
                        },
                        {
                            "evidence_id": "ev:conservation",
                            "variant_id": "v1",
                            "channel": "conservation",
                            "statement": "Single-site evolutionary prior only.",
                            "raw_features": {
                                "sites": {"39": {"effective_count": 15.13}},
                                "neff": 15.13,
                                "neff_per_length": 0.27,
                                "pseudocount_mode": "neff_scaled_uniform",
                                "pairwise_enabled": False,
                                "pairwise_eligible": False,
                                "estimated_parameters": ["pseudocount_weight"],
                            },
                            "provenance": {
                                "provider": "MSAProfileProvider",
                                "resource_sha256": "msa-hash",
                            },
                        },
                    ],
                    "provenance": [],
                }
            ],
        },
    }

    messages = build_scientist_hypothesis_messages(
        profile="scientist profile",
        sanitized_context=context,
        evidence=[],
        output_schema=HYPOTHESIS_SCHEMA,
    )

    payload = json.loads(messages[1]["content"])
    feature = payload["context"]["kg_interaction"]["packs"][0]["evidence"][0]
    assert feature["raw_features"]["sites"]["39"]["contact_count"] == 9
    assert feature["raw_features"]["resource_id"] == "rcsb:1PGB"
    assert feature["provenance"]["resource_sha256"] == "abc"
    conservation = payload["context"]["kg_interaction"]["packs"][0]["evidence"][1]
    assert conservation["raw_features"]["neff_per_length"] == 0.27
    assert conservation["raw_features"]["pairwise_enabled"] is False
    assert conservation["raw_features"]["estimated_parameters"] == ["pseudocount_weight"]
    assert "backend_coordinate_dump" not in messages[1]["content"]
    assert "private_cache_state" not in messages[1]["content"]


def test_mock_scientist_uses_critic_revision_parent_and_new_id() -> None:
    context = {
        **_context(),
        "expected_hypothesis_id": "hyp:run:r1:a1",
        "previous_hypothesis_id": "hyp:run:r1",
        "critic_revision": {
            "verdict": "REVISE",
            "summary": "Add controls and change residues.",
            "rejected_hypothesis_id": "hyp:run:r1",
            "rejected_preferred_residues": {
                "39": ["W"],
                "40": ["D"],
                "41": ["G"],
                "54": ["V"],
            },
        },
    }
    hypothesis = MockScientistLLMClient().generate_hypothesis(
        sanitized_context=context,
        evidence=[],
        output_schema=HYPOTHESIS_SCHEMA,
    )
    assert hypothesis.hypothesis_id == "hyp:run:r1:a1"
    assert hypothesis.parent_hypothesis_id == "hyp:run:r1"
    assert "Revised after critic" in hypothesis.statement


def _reflection(variant_id: str) -> dict:
    return {
        "variant_id": variant_id,
        "verdict": "support",
        "summary": "Wet evidence supports the round-specific reason.",
        "positive_findings": ["Wet validation exceeded the baseline."],
        "negative_findings": [],
        "revised_reason": "Keep the reason bounded to this round.",
        "next_round_advice": "Test matched alternatives.",
    }


def test_rethink_coverage_mismatch_retries_inside_structured_boundary() -> None:
    remote = _SequenceClient(
        [
            {"reflections": [_reflection("v1")]},
            {"reflections": [_reflection("v1"), _reflection("v2")]},
        ]
    )
    client = NativeReThinkClient.__new__(NativeReThinkClient)
    client.model = "unit-test-model"
    client.temperature = 0.0
    client.max_tokens = 1024
    client.reasoning_effort = None
    client.thinking = None
    client.profile_name = "scientific_v1"
    client.profile = "Return exact candidate coverage."
    client.profile_sha256 = "test-profile"
    client.client = remote
    client.transport = OpenAICompatibleChatTransport(remote)
    context = ReThinkContextInput.model_validate(
        {
            "run_id": "run",
            "round_id": 1,
            "visible_baseline": 0.0,
            "candidates": [{"variant_id": "v1"}, {"variant_id": "v2"}],
        }
    )

    reflections = client.reflect_round(context=context)

    assert remote.calls == 2
    assert {item.variant_id for item in reflections} == {"v1", "v2"}
