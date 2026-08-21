from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fitness_agents.agents.critic import (
    CritiqueDecisionBodyOutput,
    CritiqueDecisionOutput,
    _decision_from_payload,
)
from fitness_agents.agents.main_hypothesis_critic import sanitize_main_review
from fitness_agents.agents.output_contracts import MainSynthesisOutput
from fitness_agents.contracts.evidence_universe import (
    RoleVisibleEvidenceEntry,
    RoleVisibleEvidenceUniverse,
)
from fitness_agents.contracts.hypothesis_pipeline import (
    ChannelHypothesisOutput,
    ConservationReviewBody,
    MainReviewBody,
    PhyschemInterpretationOutput,
    PhyschemReviewBody,
    StructureReviewBody,
)
from fitness_agents.contracts.schemas import DraftBatch, Hypothesis

ROOT = Path(__file__).parents[2]


def test_rating_region_controls_critic_verdict_and_text_error_ceiling() -> None:
    base = {
        "falsification_readiness": "ready",
        "candidate_issues": [],
        "batch_level_risks": [],
        "evidence_conflicts": [],
        "unsupported_claims": [],
        "required_changes": [],
        "cited_evidence_ids": [],
        "confidence": 0.9,
        "explanation": "Bounded review.",
    }
    approved = CritiqueDecisionBodyOutput.model_validate(
        {
            **base,
            "verdict": "APPROVE",
            "rating": {
                "score": 4,
                "rationale": "No unresolved defect.",
                "suggestions": [],
                "text_errors": [],
            },
        }
    )
    assert approved.rating.score == 4
    with pytest.raises(ValidationError, match="verdict must be REVISE"):
        CritiqueDecisionBodyOutput.model_validate(
            {
                **base,
                "verdict": "APPROVE",
                "rating": {
                    "score": 3,
                    "rationale": "Repairable text defect.",
                    "suggestions": ["Correct the text defect."],
                    "text_errors": ["One statement is internally inconsistent."],
                },
            }
        )


def _backticks_between(text: str, start: str, end: str) -> set[str]:
    fragment = text.split(start, 1)[1].split(end, 1)[0]
    return set(re.findall(r"`([A-Z][A-Z0-9_]+)`", fragment))


def _resolve(node: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return root["$defs"][ref.rsplit("/", 1)[-1]]
    return node


def _property_enums(schema: dict[str, Any], property_name: str) -> set[str]:
    values: set[str] = set()

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node = _resolve(node, schema)
        properties = node.get("properties") or {}
        if property_name in properties:
            collect(properties[property_name])
        for value in node.values():
            if isinstance(value, dict):
                walk(value)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

    def collect(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node = _resolve(node, schema)
        values.update(str(item) for item in node.get("enum", ()))
        if "const" in node:
            values.add(str(node["const"]))
        for key in ("items", "anyOf", "oneOf", "allOf"):
            child = node.get(key)
            if isinstance(child, dict):
                collect(child)
            elif isinstance(child, list):
                for item in child:
                    collect(item)

    walk(schema)
    return values


CASES = (
    (
        ROOT / "src/fitness_agents/agents/profiles/subcritic/physchem_v1/SKILL.md",
        PhyschemReviewBody,
        "Issue codes:",
        "Required actions:",
        "Required actions:",
        "## Output limits",
    ),
    (
        ROOT / "src/fitness_agents/agents/profiles/subcritic/conservation_v1/SKILL.md",
        ConservationReviewBody,
        "Issue codes:",
        "Required actions:",
        "Required actions:",
        "## Output limits",
    ),
    (
        ROOT / "src/fitness_agents/agents/profiles/subcritic/structure_v1/SKILL.md",
        StructureReviewBody,
        "Issue codes:",
        "Required actions:",
        "Required actions:",
        "## Output limits",
    ),
    (
        ROOT / "src/fitness_agents/agents/profiles/critic/hypothesis_v1/SKILL.md",
        MainReviewBody,
        "Issue codes:",
        "Required actions:",
        "Required actions:",
        "## Output limits",
    ),
    (
        ROOT / "src/fitness_agents/agents/critic_profiles/scientific_v1/SKILL.md",
        CritiqueDecisionBodyOutput,
        "## 6.1 Allowed issue/risk codes",
        "Do not emit `FORMAT_INVALID`",
        "## 6. Allowed required-change actions",
        "## 6.1 Allowed issue/risk codes",
    ),
)


@pytest.mark.parametrize(
    "path,model,code_start,code_end,action_start,action_end", CASES
)
def test_skill_schema_enum_parity(
    path: Path,
    model: type,
    code_start: str,
    code_end: str,
    action_start: str,
    action_end: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    schema = model.model_json_schema()
    assert _backticks_between(text, code_start, code_end) == _property_enums(
        schema, "code"
    )
    assert _backticks_between(text, action_start, action_end) == _property_enums(
        schema, "action"
    ).union(_property_enums(schema, "required_changes"))
    assert "schema_sha256" not in text
    assert "skill_sha256" not in text


def test_batch_model_visible_contract_omits_runtime_identifiers() -> None:
    visible = CritiqueDecisionBodyOutput.model_json_schema()["properties"]
    runtime = CritiqueDecisionOutput.model_json_schema()["properties"]
    deterministic = {"decision_id", "draft_batch_id", "round_id", "review_attempt"}
    assert deterministic.isdisjoint(visible)
    assert deterministic.issubset(runtime)
    assert "hypothesis" not in visible
    assert "explanation" in visible
    assert "summary" not in visible


def test_main_critic_returns_explanation_without_a_hypothesis() -> None:
    properties = MainReviewBody.model_json_schema()["properties"]
    assert "explanation" in properties
    assert "summary" not in properties
    assert "hypothesis" not in properties


def test_critic_vocabularies_do_not_cross_role_boundaries() -> None:
    child_schemas = [
        PhyschemReviewBody.model_json_schema(),
        ConservationReviewBody.model_json_schema(),
        StructureReviewBody.model_json_schema(),
    ]
    child_codes = set().union(
        *(_property_enums(schema, "code") for schema in child_schemas)
    )
    child_actions = set().union(
        *(_property_enums(schema, "required_changes") for schema in child_schemas)
    )
    assert {"FORMAT_INVALID", "CITATION_UNKNOWN", "CROSS_CHANNEL_CONFLICT"}.isdisjoint(
        child_codes
    )
    assert {"ADD_CONTROL", "INCREASE_DIVERSITY"}.isdisjoint(child_actions)
    assert "CROSS_CHANNEL_CONFLICT" in _property_enums(
        MainReviewBody.model_json_schema(), "code"
    )
    assert {"ADD_CONTROL", "INCREASE_DIVERSITY"}.issubset(
        _property_enums(CritiqueDecisionBodyOutput.model_json_schema(), "action")
    )


def test_batch_runtime_injects_deterministic_envelope_fields() -> None:
    draft = DraftBatch(
        draft_batch_id="draft:1",
        parent_draft_batch_id=None,
        round_id=2,
        review_attempt=1,
        candidate_ids=("v1",),
        hypothesis_ids=("hyp:1",),
        prediction_snapshot_id="prediction:1",
        evidence_snapshot_id="evidence:1",
        acquisition_snapshot_id="acquisition:1",
        design_rationales=(),
        falsification_spec=None,
    )
    payload = {
        "verdict": "APPROVE",
        "falsification_readiness": "ready",
        "candidate_issues": [],
        "batch_level_risks": [],
        "evidence_conflicts": [],
        "unsupported_claims": [],
        "required_changes": [],
        "cited_evidence_ids": [],
        "confidence": 0.9,
        "summary": "Approved.",
    }
    decision = _decision_from_payload(payload, draft=draft)
    assert decision.draft_batch_id == "draft:1"
    assert decision.round_id == 2
    assert decision.review_attempt == 1
    assert decision.decision_id == "D02-01"


def test_batch_nested_models_forbid_extra_fields_and_unknown_codes() -> None:
    payload = {
        "verdict": "REVISE",
        "falsification_readiness": "needs_revision",
        "candidate_issues": [
            {
                "issue_id": "i1",
                "candidate_id": "v1",
                "scope": "sequence",
                "severity": "error",
                "code": "NOT_ALLOW_LISTED",
                "claim": "Unsupported code.",
                "evidence_ids": [],
                "conflict_ids": [],
                "suggested_action": None,
                "unexpected": True,
            }
        ],
        "batch_level_risks": [],
        "evidence_conflicts": [],
        "unsupported_claims": [],
        "required_changes": [],
        "cited_evidence_ids": [],
        "confidence": 0.5,
        "summary": "Revise.",
    }
    with pytest.raises(ValidationError) as captured:
        CritiqueDecisionBodyOutput.model_validate(payload)
    paths = {tuple(item["loc"]) for item in captured.value.errors(include_input=False)}
    assert ("candidate_issues", 0, "code") in paths
    assert ("candidate_issues", 0, "unexpected") in paths


@pytest.mark.parametrize(
    "channel,path",
    (
        (
            "physchem",
            ROOT
            / "src/fitness_agents/agents/profiles/subscientist/physchem_v1/SKILL.md",
        ),
        (
            "conservation",
            ROOT
            / "src/fitness_agents/agents/profiles/subscientist/conservation_v1/SKILL.md",
        ),
        (
            "structure",
            ROOT
            / "src/fitness_agents/agents/profiles/subscientist/structure_v1/SKILL.md",
        ),
    ),
)
def test_subscientist_skill_matches_analysis_contract_and_example(
    channel: str, path: Path
) -> None:
    text = path.read_text(encoding="utf-8")
    assert "sha256" not in text.casefold()
    if channel == "physchem":
        match = re.search(r"Example: `([^`]+)`", text)
        assert match
        example = PhyschemInterpretationOutput.model_validate_json(match.group(1))
        assert example.sample_ids == ["S01"]
        assert example.evidence_ids == []
        assert example.fact_ids == []
        assert "id_maps" in text
        assert "never invent" in text.casefold()
        assert "prefer mutation tokens" in text.casefold()
        assert "sibling batch" in text.casefold()
        assert "always keep empty" in text.casefold()
        assert "ADD_EVIDENCE_LINK" in text
        return
    match = re.search(r"Analysis-only example: `([^`]+)`", text)
    assert match
    output = ChannelHypothesisOutput.model_validate_json(match.group(1))
    assert output.channel == channel
    assert output.candidate_hypotheses == []
    folded = text.casefold()
    assert "id_maps" in text
    assert "sibling batch" in folded
    assert "mutation notation" in folded
    assert "evidence_ids: []" in text
    assert "V54C" not in text
    assert "S95" not in text


def test_main_synthesis_skill_matches_hypothesis_contract() -> None:
    path = (
        ROOT
        / "src/fitness_agents/agents/profiles/scientist/synthesis_v1/SKILL.md"
    )
    text = path.read_text(encoding="utf-8")
    compact = " ".join(text.split())
    folded = compact.casefold()
    assert "sha256" not in folded
    assert "approved_channel_analyses" in text
    assert "evidence_universe" in text
    assert "suggestions" in text
    assert "hard_residue_constraints" in text
    assert "OVERCONFIDENT" in text
    assert "visible measurement association" in folded
    assert "do not invent or wait for an assay support card" in folded
    assert "never copy" in folded
    assert "no hypothesis explanation field" in folded
    assert "soft set of alternatives" in folded
    assert "do not repair by switching to `association` and one letter per site" in folded
    assert "keep any conservation, structure" in folded
    assert "instead of rewriting the draft" in folded
    assert '"39":["L","I","V"]' in text
    assert '"39":["V"],"40":["D"]' not in text
    match = re.search(r"SYNTHESIZED example for `[^`]+`: `([^`]+)`", text)
    assert match
    example = MainSynthesisOutput.model_validate_json(match.group(1)).root
    residue_sets = example.preferred_residues
    assert any(len(residues) > 1 for residues in residue_sets.values())


def test_critic_skills_state_coupled_verdict_contract() -> None:
    paths = (
        ROOT / "src/fitness_agents/agents/profiles/critic/hypothesis_v1/SKILL.md",
        ROOT / "src/fitness_agents/agents/profiles/subcritic/physchem_v1/SKILL.md",
        ROOT / "src/fitness_agents/agents/profiles/subcritic/conservation_v1/SKILL.md",
        ROOT / "src/fitness_agents/agents/profiles/subcritic/structure_v1/SKILL.md",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "## Coupled verdict contract" in text
        assert "not a substitute for `required_changes`" in text


def test_main_critic_skill_allows_analysis_only_measurement_association() -> None:
    text = (
        ROOT / "src/fitness_agents/agents/profiles/critic/hypothesis_v1/SKILL.md"
    ).read_text(encoding="utf-8")
    folded = text.casefold()
    compact = " ".join(text.split())
    assert "analysis_only` is neither support nor a defect" in compact
    assert "named visible measurement association" in folded
    assert "scientist cannot create cards" in folded
    assert "hypothesis.statement" in text
    assert "hypothesis.expected_outcome" in text
    assert "do not inspect your own `explanation`" in folded
    assert "do not survive across rounds" in folded
    assert "multi-residue soft priors" in folded
    assert "singleton map as `untestable`" in folded
    assert "make_falsifiable" in folded
    assert "do not require a child `candidate_hypotheses`" in folded
    assert "## retry review" in folded
    assert "prior_review" in folded
    assert "do not invent a new defect code" in folded
    assert "empty child" in folded
    assert "multi-residue soft set is not" in folded


def test_structure_conservation_skills_use_generic_mutation_and_limitation_contract() -> None:
    scientist_paths = (
        ROOT / "src/fitness_agents/agents/profiles/subscientist/structure_v1/SKILL.md",
        ROOT / "src/fitness_agents/agents/profiles/subscientist/conservation_v1/SKILL.md",
    )
    critic_paths = (
        ROOT / "src/fitness_agents/agents/profiles/subcritic/structure_v1/SKILL.md",
        ROOT / "src/fitness_agents/agents/profiles/subcritic/conservation_v1/SKILL.md",
    )
    forbidden = ("V54C", "S95", "V39A", "G41D")
    for path in scientist_paths:
        text = path.read_text(encoding="utf-8")
        folded = text.casefold()
        compact = " ".join(text.split())
        assert "mutation notation" in folded
        assert "sibling batch" in folded
        assert "evidence_ids: []" in compact
        for token in forbidden:
            assert token not in text
    for path in critic_paths:
        text = path.read_text(encoding="utf-8")
        folded = text.casefold()
        assert "mutation notation" in folded
        assert "sample-label mismatch" in folded
        assert "empty `evidence_ids` on `limitation`" in folded
        assert "do not emit `finding_unsupported` or `add_evidence_link`" in folded
        for token in forbidden:
            assert token not in text
    structure_critic = (
        ROOT / "src/fitness_agents/agents/profiles/subcritic/structure_v1/SKILL.md"
    ).read_text(encoding="utf-8")
    assert "ACKNOWLEDGE_MISSING_COORDINATES" in structure_critic
    assert "different sample or mutation token" in structure_critic
    conservation_critic = (
        ROOT / "src/fitness_agents/agents/profiles/subcritic/conservation_v1/SKILL.md"
    ).read_text(encoding="utf-8")
    assert "COORDINATES_MISSING" not in conservation_critic
    assert "ACKNOWLEDGE_MISSING_COORDINATES" not in conservation_critic


def test_physchem_critic_skill_accepts_empty_interpretation_citations() -> None:
    text = (
        ROOT / "src/fitness_agents/agents/profiles/subcritic/physchem_v1/SKILL.md"
    ).read_text(encoding="utf-8")
    folded = text.casefold()
    compact = " ".join(text.split())
    assert "empty interpretation" in folded
    assert "Do not emit `FINDING_UNSUPPORTED` or `ADD_EVIDENCE_LINK`" in compact
    assert "mutation tokens" in folded
    assert "sample-label" in folded


def _soft_set_hypothesis() -> Hypothesis:
    return Hypothesis(
        hypothesis_id="H02-02",
        statement=(
            "Test a narrowed four-position soft directional prior with explicit "
            "alternative sets from visible measured-fitness associations."
        ),
        preferred_residues={
            39: ("I", "L", "C"),
            40: ("W", "Y", "H"),
            41: ("G", "A"),
            54: ("A", "F", "C"),
        },
        evidence_ids=("ev:kg:1",),
        expected_outcome="The selected batch median exceeds the pre-round visible median.",
        falsification_criterion="Batch median lift versus pre-round observations.",
        claim_modality="directional_prior",
    )


def _universe(*evidence_ids: str) -> RoleVisibleEvidenceUniverse:
    return RoleVisibleEvidenceUniverse(
        role="main_critic",
        entries=tuple(
            RoleVisibleEvidenceEntry(evidence_id=item, origins=("test",))
            for item in evidence_ids
        ),
    )


def test_sanitize_main_review_drops_empty_child_and_soft_set_untestable() -> None:
    hypothesis = _soft_set_hypothesis()
    universe = _universe("ev:kg:1")
    payload = {
        "review_scope": "main",
        "verdict": "REVISE",
        "rating": {
            "score": 3,
            "rationale": "Repairable falsifiability defect.",
            "suggestions": ["Add a child candidate or uniquely named residue."],
            "text_errors": [],
        },
        "issues": [
            {
                "code": "UNTESTABLE",
                "severity": "error",
                "message": (
                    "The hypothesis states no child candidate hypotheses and the "
                    "approved channel analyses also contain no candidate hypotheses."
                ),
                "evidence_ids": [],
            },
            {
                "code": "UNTESTABLE",
                "severity": "warning",
                "message": (
                    "The preferred residue sets are not uniquely named by any cited "
                    "evidence card."
                ),
                "evidence_ids": [],
            },
        ],
        "required_changes": ["MAKE_FALSIFIABLE"],
        "cited_evidence_ids": ["ev:kg:1"],
        "explanation": "The soft prior is bounded but treated as untestable.",
    }
    sanitized = sanitize_main_review(
        payload, hypothesis=hypothesis, evidence_universe=universe
    )
    assert sanitized["issues"] == []
    assert sanitized["verdict"] == "APPROVE"
    assert sanitized["required_changes"] == []
    assert sanitized["rating"]["score"] == 4


def test_sanitize_main_review_keeps_singleton_untestable() -> None:
    hypothesis = Hypothesis(
        hypothesis_id="H01-00",
        statement="Test the singleton association V39I, D40W, G41A, V54C.",
        preferred_residues={39: ("I",), 40: ("W",), 41: ("A",), 54: ("C",)},
        evidence_ids=("ev:kg:1",),
        expected_outcome="The selected batch median exceeds the pre-round visible median.",
        falsification_criterion="Batch median lift versus pre-round observations.",
        claim_modality="association",
    )
    payload = {
        "review_scope": "main",
        "verdict": "REVISE",
        "rating": {
            "score": 3,
            "rationale": "Singleton map is not contrastable.",
            "suggestions": ["Expand each site to a soft alternative set."],
            "text_errors": [],
        },
        "issues": [
            {
                "code": "UNTESTABLE",
                "severity": "error",
                "message": "The singleton all-site map cannot be contrasted in the visible design space.",
                "evidence_ids": [],
            }
        ],
        "required_changes": ["MAKE_FALSIFIABLE"],
        "cited_evidence_ids": ["ev:kg:1"],
        "explanation": "Each site is a single letter without a uniquely named card.",
    }
    sanitized = sanitize_main_review(
        payload, hypothesis=hypothesis, evidence_universe=_universe("ev:kg:1")
    )
    assert sanitized["verdict"] == "REVISE"
    assert sanitized["issues"][0]["code"] == "UNTESTABLE"
    assert sanitized["required_changes"] == ["MAKE_FALSIFIABLE"]
