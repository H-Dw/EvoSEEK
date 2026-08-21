from __future__ import annotations

from fitness_agents.agents.main_hypothesis_critic import sanitize_main_review
from fitness_agents.agents.output_contracts import validate_main_synthesis_payload
from fitness_agents.agents.short_ids import ShortIdMap
from fitness_agents.contracts.evidence_universe import (
    RoleVisibleEvidenceEntry,
    RoleVisibleEvidenceUniverse,
)
from fitness_agents.contracts.hypothesis_pipeline import MainSynthesisEvidenceCard
from fitness_agents.contracts.schemas import Hypothesis


def _universe(*evidence_ids: str) -> RoleVisibleEvidenceUniverse:
    return RoleVisibleEvidenceUniverse(
        role="main_scientist_and_critic",
        entries=tuple(
            RoleVisibleEvidenceEntry(
                evidence_id=evidence_id,
                origins=("evidence",),
            )
            for evidence_id in evidence_ids
        ),
    )


def _hypothesis(**overrides: object) -> Hypothesis:
    payload: dict[str, object] = {
        "hypothesis_id": "H01-00",
        "statement": (
            "Test the bounded four-position visible measurement association "
            "V39C, D40Y, G41G, V54V. Conservation log-odds are treated as an "
            "evolutionary prior."
        ),
        "preferred_residues": {39: ("C",), 40: ("Y",), 41: ("G",), 54: ("V",)},
        "evidence_ids": ("ev:kg:1", "ev:cons:1"),
        "expected_outcome": (
            "The selected batch median exceeds the pre-round visible median; "
            "conservation constraints reduce confidence."
        ),
        "falsification_criterion": "Runtime-rendered from the typed template.",
        "claim_modality": "association",
        "preference_strength_by_position": {
            39: "exploratory",
            40: "exploratory",
            41: "exploratory",
            54: "exploratory",
        },
    }
    payload.update(overrides)
    return Hypothesis(**payload)


def _card(**overrides: object) -> MainSynthesisEvidenceCard:
    payload: dict[str, object] = {
        "evidence_id": "ev:kg:1",
        "atomic_statement": "context-bound residue observation score=0.757; support=4",
        "channel": "kg",
        "contribution": "analysis_only",
        "polarity": "neutral",
        "applicability": "in_domain",
        "confidence": 0.37,
        "quality_status": "ok",
        "warnings": ("descriptive_association_not_causal",),
    }
    payload.update(overrides)
    return MainSynthesisEvidenceCard(**payload)


def _revise_payload(*, issues: list[dict], changes: list[str]) -> dict:
    return {
        "review_scope": "main",
        "verdict": "REVISE",
        "rating": {
            "score": 2,
            "rationale": "Repairable synthesis defects.",
            "suggestions": ["Add an assay support card or lower the claim."],
            "text_errors": [],
        },
        "issues": issues,
        "required_changes": changes,
        "cited_evidence_ids": ["ev:kg:1"],
        "explanation": "The cited cards are analysis_only and omit assay fitness.",
    }


def test_short_id_map_strips_prior_round_aliases_without_touching_canonical_ids() -> None:
    id_map = ShortIdMap.build(("ev:kg:1", "ev:cons:1"), prefix="E")
    text = "Cite E01, ignore E33/E27, and keep E1:kg:sha256:deadbeef."
    assert id_map.unknown_aliases_in_text(text) == ("E33", "E27")
    stripped = id_map.strip_unknown_aliases_in_text(text)
    assert "E01" in stripped
    assert "E33" not in stripped
    assert "E27" not in stripped
    assert "E1:kg:sha256:deadbeef" in stripped


def test_sanitize_main_review_approves_analysis_only_measurement_association() -> None:
    hypothesis = _hypothesis()
    universe = _universe("ev:kg:1", "ev:cons:1")
    cards = (
        _card(),
        _card(
            evidence_id="ev:cons:1",
            atomic_statement="MSA single-site log-odds=-3.170; evolutionary prior",
            channel="conservation",
            contribution="constraint_counterevidence",
            confidence=0.0,
            warnings=("evolutionary_profile_not_assay_fitness",),
        ),
    )
    sanitized = sanitize_main_review(
        _revise_payload(
            issues=[
                {
                    "code": "UNSUPPORTED_SYNTHESIS",
                    "severity": "error",
                    "message": (
                        "The central directional claim is unsupported. Cited cards are "
                        "analysis_only with neutral polarity and no assay fitness card."
                    ),
                    "evidence_ids": ["ev:kg:1"],
                },
                {
                    "code": "COUNTEREVIDENCE_IGNORED",
                    "severity": "warning",
                    "message": (
                        "The explanation does not address conservation log-odds "
                        "counterevidence."
                    ),
                    "evidence_ids": ["ev:cons:1"],
                },
                {
                    "code": "OVERCONFIDENT",
                    "severity": "warning",
                    "message": (
                        "The expected outcome is not supported by any visible evidence "
                        "and exceeds the available confidence of analysis_only cards."
                    ),
                    "evidence_ids": [],
                },
            ],
            changes=["ADD_EXPLANATION", "ADD_COUNTEREVIDENCE", "LOWER_CONFIDENCE"],
        ),
        hypothesis=hypothesis,
        evidence_universe=universe,
        evidence_cards=cards,
    )
    assert sanitized["verdict"] == "APPROVE"
    assert sanitized["required_changes"] == []
    assert sanitized["issues"] == []
    assert sanitized["rating"]["score"] == 4


def test_sanitize_main_review_drops_stale_short_ids_and_missing_assay_demands() -> None:
    hypothesis = _hypothesis(
        statement="Test the bounded association from S107 using aggregate evidence E33/E27.",
        expected_outcome="The batch median exceeds the pre-round visible median.",
    )
    universe = _universe("ev:kg:1")
    sanitized = sanitize_main_review(
        _revise_payload(
            issues=[
                {
                    "code": "UNSUPPORTED_SYNTHESIS",
                    "severity": "error",
                    "message": (
                        "The hypothesis references aggregate evidence E33/E27, which "
                        "are not present in the allowed evidence universe."
                    ),
                    "evidence_ids": [],
                }
            ],
            changes=["NARROW_CLAIM", "ADD_EXPLANATION"],
        ),
        hypothesis=hypothesis,
        evidence_universe=universe,
        evidence_cards=(_card(),),
    )
    assert sanitized["verdict"] == "APPROVE"
    assert sanitized["issues"] == []


def test_sanitize_main_review_keeps_counterevidence_when_statement_omits_it() -> None:
    hypothesis = _hypothesis(
        statement="Test V39C as a directional prior from a new inference.",
        expected_outcome="The batch median exceeds the pre-round visible median.",
        claim_modality="directional_prior",
        evidence_ids=("ev:kg:1",),
    )
    universe = _universe("ev:kg:1", "ev:cons:1")
    cards = (
        _card(),
        _card(
            evidence_id="ev:cons:1",
            atomic_statement="MSA log-odds=-5.164 against V39C",
            channel="conservation",
            contribution="constraint_counterevidence",
            confidence=0.0,
        ),
    )
    sanitized = sanitize_main_review(
        _revise_payload(
            issues=[
                {
                    "code": "COUNTEREVIDENCE_IGNORED",
                    "severity": "error",
                    "message": "Visible conservation counterevidence is omitted.",
                    "evidence_ids": ["ev:cons:1"],
                }
            ],
            changes=["ADD_COUNTEREVIDENCE"],
        ),
        hypothesis=hypothesis,
        evidence_universe=universe,
        evidence_cards=cards,
    )
    assert sanitized["verdict"] == "REVISE"
    assert sanitized["required_changes"] == ["ADD_COUNTEREVIDENCE"]
    assert sanitized["issues"][0]["code"] == "COUNTEREVIDENCE_IGNORED"


def test_sanitize_main_review_keeps_residue_hardness_overconfident() -> None:
    hypothesis = _hypothesis(
        statement="V39 must occupy Cys. This is a visible measurement association.",
        claim_modality="directional_prior",
    )
    universe = _universe("ev:kg:1")
    sanitized = sanitize_main_review(
        _revise_payload(
            issues=[
                {
                    "code": "OVERCONFIDENT",
                    "severity": "error",
                    "message": (
                        "Verbal residue hardness V39 must is present while "
                        "hard_residue_constraints is empty."
                    ),
                    "evidence_ids": [],
                }
            ],
            changes=["LOWER_CONFIDENCE"],
        ),
        hypothesis=hypothesis,
        evidence_universe=universe,
        evidence_cards=(_card(),),
    )
    assert sanitized["verdict"] == "REVISE"
    assert sanitized["required_changes"] == ["LOWER_CONFIDENCE"]
    assert sanitized["issues"][0]["code"] == "OVERCONFIDENT"


def test_sanitize_main_review_keeps_true_unsupported_directional_claim() -> None:
    hypothesis = _hypothesis(
        statement="Prefer V39C because this is a new inference that still requires testing.",
        expected_outcome="Fitness will increase.",
        claim_modality="directional_prior",
        evidence_ids=("ev:kg:1",),
    )
    universe = _universe("ev:kg:1")
    sanitized = sanitize_main_review(
        _revise_payload(
            issues=[
                {
                    "code": "UNSUPPORTED_SYNTHESIS",
                    "severity": "error",
                    "message": (
                        "The central directional claim has no supporting card or "
                        "approved child candidate."
                    ),
                    "evidence_ids": [],
                }
            ],
            changes=["NARROW_CLAIM"],
        ),
        hypothesis=hypothesis,
        evidence_universe=universe,
        evidence_cards=(_card(),),
    )
    assert sanitized["verdict"] == "REVISE"
    assert sanitized["required_changes"] == ["NARROW_CLAIM"]
    assert sanitized["issues"][0]["code"] == "UNSUPPORTED_SYNTHESIS"


def test_validate_main_synthesis_strips_prior_round_short_ids() -> None:
    dumped = validate_main_synthesis_payload(
        {
            "outcome": "SYNTHESIZED_HYPOTHESIS",
            "statement": (
                "Test the bounded association supported by E01, not prior-round E33."
            ),
            "claim_modality": "association",
            "preferred_residues": {
                "39": ["C"],
                "40": ["Y"],
                "41": ["G"],
                "54": ["V"],
            },
            "preference_strength_by_position": {
                "39": "exploratory",
                "40": "exploratory",
                "41": "exploratory",
                "54": "exploratory",
            },
            "evidence_ids": ["ev:kg:1"],
            "expected_outcome": "The preregistered comparison separates the direction.",
            "falsification_criterion": "Runtime-rendered from the typed template.",
            "hard_residue_constraints": {},
            "falsification_template": {
                "detector": "batch_median_lift",
                "target_relation": "selected_batch",
                "comparator_relation": "pre_round_visible_observations",
                "operator": "greater",
                "threshold_source": "zero_lift",
                "min_observations": "selected_batch_size",
                "missing_data_policy": "INCONCLUSIVE",
                "reduction_policy": "primary_contradiction_first_v1",
            },
        },
        expected_hypothesis_id="H01-00",
        expected_parent_hypothesis_id=None,
        allowed_evidence_ids=frozenset({"ev:kg:1"}),
        expected_positions=(39, 40, 41, 54),
        allowed_positions=None,
        max_positions=None,
    )
    assert "E01" in dumped["statement"]
    assert "E33" not in dumped["statement"]
    assert dumped["evidence_ids"] == ["ev:kg:1"]
