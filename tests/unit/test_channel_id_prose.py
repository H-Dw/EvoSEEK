from __future__ import annotations

from fitness_agents.agents.short_ids import ShortIdMap, rewrite_exact_ids
from fitness_agents.agents.subcritic import sanitize_channel_review, validate_subcritic_review
from fitness_agents.agents.subscientist import rewrite_channel_analysis_prose
from fitness_agents.contracts.hypothesis_pipeline import (
    ChannelAnalysisOutput,
    ChannelEvidenceInput,
)

SIBLING_EVIDENCE = "E0:structure:sha256:" + "aa" * 32
BATCH2_EVIDENCE = "E0:structure:sha256:" + "bb" * 32
BATCH1_SAMPLE = "S05"
BATCH2_SAMPLE = "S95"


def _analysis(
    *,
    channel: str = "structure",
    sample_label: str = "S01",
    evidence_label: str = "E01",
    evidence_id: str = BATCH2_EVIDENCE,
    observation: str | None = None,
    limitation: str | None = None,
) -> ChannelAnalysisOutput:
    observation_statement = observation or (
        f"{sample_label} has a coordinate-backed local environment citing {evidence_label}."
    )
    limitation_statement = limitation or (
        f"{sample_label} has no coordinates at the requested positions."
    )
    return ChannelAnalysisOutput(
        analysis_id="A-ST-01",
        channel=channel,
        analysis_summary=f"{sample_label} has a bounded static environment.",
        findings=[
            {
                "finding_id": "F-B001-F01",
                "kind": "OBSERVATION",
                "statement": observation_statement,
                "evidence_ids": [evidence_id],
                "fact_ids": [],
                "confidence": "medium",
            },
            {
                "finding_id": "F-B001-F04",
                "kind": "LIMITATION",
                "statement": limitation_statement,
                "evidence_ids": [],
                "fact_ids": [],
                "confidence": "high",
            },
        ],
        candidate_hypotheses=[],
        evidence_ids=[evidence_id],
        fact_ids=[],
        counterevidence=[f"{sample_label} does not support a dynamic claim."],
        uncertainty=f"{sample_label} missing coordinates remain a LIMITATION.",
    )


def _structure_context(
    *,
    sample_ids: tuple[str, ...] = (BATCH2_SAMPLE, BATCH1_SAMPLE),
    mutations: tuple[str, ...] = ("V39A", "D40A"),
    channel: str = "structure",
) -> ChannelEvidenceInput:
    evidence_ids = tuple(
        f"E0:{channel}:sha256:{index:064d}" for index in range(1, len(sample_ids) + 1)
    )
    return ChannelEvidenceInput(
        run_id="run:channel-id-prose",
        round_id=1,
        channel=channel,
        task="summarize channel-local feature evidence",
        mutable_positions=(39, 40),
        wild_type_sites="VD",
        sample_map={sample_id: mutation for sample_id, mutation in zip(sample_ids, mutations)},
        visible_observations=tuple(
            {
                "sample_id": sample_id,
                "mutation_notation": mutation,
                "residues_by_position": {str(int(mutation[1:-1])): mutation[-1]},
                "evidence_ids": (evidence_id,),
                "feature_values": {evidence_id: {"available": True}},
                "descriptor_facts": (),
            }
            for sample_id, mutation, evidence_id in zip(sample_ids, mutations, evidence_ids)
        ),
        evidence=tuple(
            {
                "evidence_id": evidence_id,
                "variant_id": sample_id,
                "channel": channel,
                "statement": f"{sample_id} has a bounded {channel} card.",
            }
            for sample_id, evidence_id in zip(sample_ids, evidence_ids)
        ),
    )


def _review_payload(
    *,
    channel: str,
    sample_ids: tuple[str, ...],
    issues: list[dict[str, object]],
    required_changes: list[str],
    score: int = 3,
) -> dict[str, object]:
    return {
        "review_scope": channel,
        "verdict": "REVISE",
        "rating": {
            "score": score,
            "rationale": "Repairable channel defects remain.",
            "suggestions": ["Repair the cited finding support."],
            "text_errors": [],
        },
        "issues": issues,
        "required_changes": required_changes,
        "cited_evidence_ids": [],
        "summary": "The analysis still needs citation or coordinate repairs.",
        "sample_reviews": [
            {
                "sample_id": sample_id,
                "feature_analysis": "Bounded channel features are visible.",
                "critic_explanation": "Sample-local review of the visible cards.",
            }
            for sample_id in sample_ids
        ],
    }


def test_two_batch_expand_uses_this_batch_maps_not_sibling_aliases() -> None:
    batch1_samples = ShortIdMap.build((BATCH1_SAMPLE,), prefix="S")
    batch1_evidence = ShortIdMap.build((SIBLING_EVIDENCE,), prefix="E")
    batch2_samples = ShortIdMap.build((BATCH2_SAMPLE,), prefix="S")
    batch2_evidence = ShortIdMap.build((BATCH2_EVIDENCE,), prefix="E")
    assert batch1_samples.alias_to_value["S01"] == BATCH1_SAMPLE
    assert batch2_samples.alias_to_value["S01"] == BATCH2_SAMPLE
    assert batch1_evidence.alias_to_value["E01"] == SIBLING_EVIDENCE
    assert batch2_evidence.alias_to_value["E01"] == BATCH2_EVIDENCE

    raw = _analysis()
    expanded = rewrite_channel_analysis_prose(
        raw, batch2_samples, batch2_evidence, mode="expand"
    )
    statement = expanded.findings[0].statement
    assert BATCH2_SAMPLE in statement
    assert BATCH2_EVIDENCE in statement
    assert "S01" not in statement
    assert "E01" not in statement
    assert BATCH1_SAMPLE not in statement
    assert SIBLING_EVIDENCE not in statement
    assert BATCH2_SAMPLE in expanded.analysis_summary
    assert BATCH2_SAMPLE in expanded.uncertainty
    assert BATCH2_SAMPLE in expanded.counterevidence[0]
    sibling_expanded = rewrite_channel_analysis_prose(
        raw, batch1_samples, batch1_evidence, mode="expand"
    )
    assert BATCH1_SAMPLE in sibling_expanded.findings[0].statement
    assert BATCH2_SAMPLE not in sibling_expanded.findings[0].statement


def test_critic_collapse_rewrites_canonical_ids_to_request_aliases() -> None:
    batch2_samples = ShortIdMap.build((BATCH2_SAMPLE,), prefix="S")
    batch2_evidence = ShortIdMap.build((BATCH2_EVIDENCE,), prefix="E")
    expanded = rewrite_channel_analysis_prose(
        _analysis(), batch2_samples, batch2_evidence, mode="expand"
    )
    extra_evidence = tuple(
        f"E0:structure:sha256:{index:064d}" for index in range(1, 9)
    )
    critic_samples = ShortIdMap.build((BATCH1_SAMPLE, BATCH2_SAMPLE), prefix="S")
    critic_evidence = ShortIdMap.build((*extra_evidence, BATCH2_EVIDENCE), prefix="E")
    encoded = rewrite_exact_ids(
        expanded.model_dump(mode="json"), critic_evidence, critic_samples
    )
    collapsed = rewrite_channel_analysis_prose(
        ChannelAnalysisOutput.model_validate(encoded),
        critic_samples,
        critic_evidence,
        mode="collapse",
    )
    evidence_alias = critic_evidence.encode(BATCH2_EVIDENCE)
    sample_alias = critic_samples.encode(BATCH2_SAMPLE)
    assert evidence_alias == "E09"
    assert sample_alias == "S02"
    assert collapsed.evidence_ids == [evidence_alias]
    assert collapsed.findings[0].evidence_ids == [evidence_alias]
    statement = collapsed.findings[0].statement
    assert evidence_alias in statement
    assert sample_alias in statement
    assert BATCH2_EVIDENCE not in statement
    assert BATCH2_SAMPLE not in statement


def test_empty_limitation_citation_revise_promotes_to_approve() -> None:
    context = _structure_context()
    hypothesis = _analysis(
        observation="S95 (V39A) has a coordinate-backed local environment.",
        limitation="S95 (V39A) has no coordinates at the requested positions.",
        evidence_id=context.evidence[0]["evidence_id"],
    )
    payload = _review_payload(
        channel="structure",
        sample_ids=tuple(item.sample_id for item in context.visible_observations),
        issues=[
            {
                "severity": "error",
                "code": "FINDING_UNSUPPORTED",
                "message": (
                    "LIMITATION F-B001-F04 has empty evidence_ids and is therefore uncited."
                ),
                "evidence_ids": [],
            },
            {
                "severity": "error",
                "code": "COORDINATES_MISSING",
                "message": (
                    "Findings reference sample IDs S01 and S02, which are not present "
                    "in the sample_map."
                ),
                "evidence_ids": [],
            },
        ],
        required_changes=["ADD_EVIDENCE_LINK", "ACKNOWLEDGE_MISSING_COORDINATES"],
    )
    sanitized = sanitize_channel_review(
        payload, context=context, hypothesis=hypothesis
    )
    assert sanitized["verdict"] == "APPROVE"
    assert sanitized["required_changes"] == []
    assert sanitized["rating"]["score"] == 4
    assert sanitized["issues"] == []

    validated = validate_subcritic_review(
        payload, context=context, hypothesis=hypothesis
    )
    assert validated["verdict"] == "APPROVE"
    assert validated["required_changes"] == []


def test_conservation_empty_limitation_citation_promotes_to_approve() -> None:
    context = _structure_context(channel="conservation")
    hypothesis = _analysis(
        channel="conservation",
        observation="S95 (V39A) has bounded single-site conservation support.",
        limitation="Pairwise analysis is ineligible in the supplied result.",
        evidence_id=context.evidence[0]["evidence_id"],
    )
    payload = _review_payload(
        channel="conservation",
        sample_ids=tuple(item.sample_id for item in context.visible_observations),
        issues=[
            {
                "severity": "error",
                "code": "FINDING_UNSUPPORTED",
                "message": "The LIMITATION finding has empty evidence_ids and missing citations.",
                "evidence_ids": [],
            }
        ],
        required_changes=["ADD_EVIDENCE_LINK"],
    )
    sanitized = sanitize_channel_review(
        payload, context=context, hypothesis=hypothesis
    )
    assert sanitized["verdict"] == "APPROVE"
    assert sanitized["required_changes"] == []
    assert sanitized["issues"] == []


def test_sanitizer_keeps_observation_with_unseen_mutation_token() -> None:
    context = _structure_context()
    hypothesis = _analysis(
        observation="G41D has a packed core that is not on the visible cards.",
        limitation="S95 (V39A) has no coordinates at the requested positions.",
        evidence_id=context.evidence[0]["evidence_id"],
    )
    payload = _review_payload(
        channel="structure",
        sample_ids=tuple(item.sample_id for item in context.visible_observations),
        issues=[
            {
                "severity": "error",
                "code": "FINDING_UNSUPPORTED",
                "message": (
                    "OBSERVATION F-B001-F01 cites mutation token G41D, which is not on "
                    "the supporting observation cards."
                ),
                "evidence_ids": [],
            }
        ],
        required_changes=["LOWER_CONFIDENCE"],
    )
    sanitized = sanitize_channel_review(
        payload, context=context, hypothesis=hypothesis
    )
    assert sanitized["verdict"] == "REVISE"
    assert sanitized["required_changes"] == ["LOWER_CONFIDENCE"]
    assert sanitized["issues"][0]["code"] == "FINDING_UNSUPPORTED"
    assert "G41D" in sanitized["issues"][0]["message"]
