from __future__ import annotations

from fitness_agents.agents.short_ids import ShortIdMap
from fitness_agents.agents.subcritic import (
    sanitize_physchem_review,
    validate_subcritic_review,
)
from fitness_agents.contracts.hypothesis_pipeline import (
    ChannelAnalysisOutput,
    ChannelEvidenceInput,
)


def _hypothesis() -> ChannelAnalysisOutput:
    return ChannelAnalysisOutput(
        analysis_id="A-PC-R01-B001",
        channel="physchem",
        analysis_summary="Bounded descriptor shifts.",
        findings=[
            {
                "finding_id": "F01",
                "kind": "OBSERVATION",
                "statement": "S95 (V54C): hydropathy delta -1.7.",
                "evidence_ids": ["ev:pc:0"],
                "fact_ids": ["D001"],
                "confidence": "high",
            },
            {
                "finding_id": "F02",
                "kind": "INTERPRETATION",
                "statement": "V54C (S95) reduces hydropathy.",
                "evidence_ids": [],
                "fact_ids": [],
                "confidence": "low",
            },
        ],
        candidate_hypotheses=[],
        evidence_ids=["ev:pc:0"],
        fact_ids=["D001"],
        counterevidence=[],
        uncertainty="Descriptor direction does not establish assay performance.",
    )


def _context() -> ChannelEvidenceInput:
    return ChannelEvidenceInput(
        run_id="run:physchem-critic",
        round_id=1,
        channel="physchem",
        task="summarize channel-local feature evidence",
        mutable_positions=(39, 40, 41, 54),
        wild_type_sites="VDGV",
        sample_map={"S95": "V54C"},
        visible_observations=(
            {
                "sample_id": "S95",
                "mutation_notation": "V54C",
                "residues_by_position": {"54": "C"},
                "evidence_ids": ("ev:pc:0",),
                "feature_values": {"ev:pc:0": {"hydropathy_delta": -1.7}},
                "descriptor_facts": (
                    {
                        "fact_id": "D001",
                        "evidence_id": "ev:pc:0",
                        "sample_id": "S95",
                        "position": 54,
                        "from_residue": "V",
                        "to_residue": "C",
                        "descriptor": "hydropathy_delta",
                        "delta": -1.7,
                    },
                ),
            },
        ),
        evidence=(
            {
                "evidence_id": "ev:pc:0",
                "variant_id": "S95",
                "channel": "physchem",
                "statement": "V54C hydropathy delta -1.7.",
            },
        ),
    )


def _review_payload(*, issues: list[dict], required_changes: list[str], score: int = 3):
    return {
        "review_scope": "physchem",
        "verdict": "REVISE" if score < 4 else "APPROVE",
        "rating": {
            "score": score,
            "rationale": "Repairable physicochemical defect.",
            "suggestions": ["Repair the cited interpretation defect."],
            "text_errors": [],
        },
        "issues": issues,
        "required_changes": required_changes,
        "cited_evidence_ids": ["ev:pc:0"],
        "summary": "Review the visible physicochemical analysis.",
        "sample_reviews": [
            {
                "sample_id": "S95",
                "feature_analysis": "Hydropathy decreases for V54C.",
                "critic_explanation": "Observation card is bounded.",
            }
        ],
    }


def test_short_id_map_expands_in_prose_but_decode_does_not() -> None:
    mapping = ShortIdMap.build(("S95",), prefix="S")
    assert mapping.decode("V54C (S01)") == "V54C (S01)"
    assert mapping.expand_aliases_in_text("V54C (S01)") == "V54C (S95)"
    mixed = ShortIdMap({"S1": "alpha", "S10": "beta"}, prefix="S")
    assert mixed.expand_aliases_in_text("see S10 then S1") == "see beta then alpha"


def test_sanitize_promotes_when_only_empty_interpretation_citations_remain() -> None:
    payload = _review_payload(
        issues=[
            {
                "code": "FINDING_UNSUPPORTED",
                "severity": "error",
                "message": "INTERPRETATION has empty fact_ids and evidence_ids.",
                "evidence_ids": [],
            }
        ],
        required_changes=["ADD_EVIDENCE_LINK"],
    )

    sanitized = sanitize_physchem_review(payload, hypothesis=_hypothesis())

    assert sanitized["verdict"] == "APPROVE"
    assert sanitized["rating"]["score"] == 4
    assert sanitized["required_changes"] == []
    assert sanitized["issues"] == []


def test_sanitize_keeps_residue_direction_and_lower_confidence() -> None:
    payload = _review_payload(
        issues=[
            {
                "code": "FINDING_UNSUPPORTED",
                "severity": "error",
                "message": "INTERPRETATION has empty fact_ids.",
                "evidence_ids": [],
            },
            {
                "code": "RESIDUE_DIRECTION_UNSUPPORTED",
                "severity": "error",
                "message": "G41Q hydropathy sign +3.1 contradicts OBSERVATION -3.1.",
                "evidence_ids": ["ev:pc:0"],
            },
        ],
        required_changes=["ADD_EVIDENCE_LINK", "LOWER_CONFIDENCE"],
    )

    sanitized = sanitize_physchem_review(payload, hypothesis=_hypothesis())

    assert sanitized["verdict"] == "REVISE"
    assert sanitized["rating"]["score"] == 3
    assert sanitized["required_changes"] == ["LOWER_CONFIDENCE"]
    assert [issue["code"] for issue in sanitized["issues"]] == [
        "RESIDUE_DIRECTION_UNSUPPORTED"
    ]


def test_sanitize_injects_lower_confidence_when_chemical_error_would_leave_revise_empty() -> None:
    payload = _review_payload(
        issues=[
            {
                "code": "RESIDUE_DIRECTION_UNSUPPORTED",
                "severity": "error",
                "message": "G41Q hydropathy sign +3.1 contradicts OBSERVATION -3.1.",
                "evidence_ids": ["ev:pc:0"],
            }
        ],
        required_changes=["ADD_EVIDENCE_LINK"],
    )

    sanitized = sanitize_physchem_review(payload, hypothesis=_hypothesis())

    assert sanitized["verdict"] == "REVISE"
    assert sanitized["required_changes"] == ["LOWER_CONFIDENCE"]
    assert sanitized["issues"][0]["code"] == "RESIDUE_DIRECTION_UNSUPPORTED"


def test_validate_subcritic_review_enforces_empty_interpretation_citation_contract() -> None:
    payload = _review_payload(
        issues=[
            {
                "code": "FINDING_UNSUPPORTED",
                "severity": "error",
                "message": "INTERPRETATION citations are empty.",
                "evidence_ids": [],
            }
        ],
        required_changes=["ADD_EVIDENCE_LINK"],
    )

    dumped = validate_subcritic_review(
        payload, context=_context(), hypothesis=_hypothesis()
    )

    assert dumped["verdict"] == "APPROVE"
    assert dumped["rating"]["score"] == 4
    assert dumped["required_changes"] == []
    assert dumped["issues"] == []
