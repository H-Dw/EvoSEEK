from __future__ import annotations

import pytest
from pydantic import ValidationError

from fitness_agents.agents.output_contracts import HYPOTHESIS_TEXT_MAX, HypothesisBodyOutput
from fitness_agents.contracts.hypothesis_pipeline import (
    CANDIDATE_PROSE_MAX,
    CRITIC_EXPLANATION_MAX,
    CRITIC_RATIONALE_MAX,
    CriticRatingRegion,
    MainReviewBody,
)


_OBSERVED_EXPLANATION = (
    "The hypothesis is a bounded, falsifiable association claim with a clear expected "
    "outcome and falsification criterion. However, it overstates support by citing "
    "analysis_only evidence as if it were supporting, and it omits visible "
    "constraint_counterevidence from conservation and structure channels. The claim "
    "strength is exploratory, but the statement uses 'supported by measured aggregate "
    "E33' without a visible supporting card, and the falsification criterion uses 'must' "
    "language that implies a stronger commitment than the exploratory preference "
    "strength. The hypothesis should be revised to accurately reflect the visible "
    "evidence and soften the falsification criterion."
)
_OBSERVED_RATIONALE = (
    "The hypothesis is a bounded, falsifiable association claim with a clear expected "
    "outcome and falsification criterion. However, it overstates support by citing "
    "analysis_only evidence as if it were supporting, and it omits visible "
    "constraint_counterevidence from conservation and structure channels. The claim "
    "strength is exploratory, but the statement uses 'supported by measured aggregate "
    "E33' without a visible supporting card, and the falsification criterion uses 'must' "
    "language that implies a stronger commitment than the exploratory preference "
    "strength."
)


def test_observed_main_critic_prose_is_within_raised_caps() -> None:
    assert len(_OBSERVED_EXPLANATION) > 600
    assert len(_OBSERVED_RATIONALE) > 400
    assert CRITIC_EXPLANATION_MAX == 2000
    assert CRITIC_RATIONALE_MAX == 1200
    assert len(_OBSERVED_EXPLANATION) <= CRITIC_EXPLANATION_MAX
    assert len(_OBSERVED_RATIONALE) <= CRITIC_RATIONALE_MAX

    rating = CriticRatingRegion.model_validate(
        {
            "score": 3,
            "rationale": _OBSERVED_RATIONALE,
            "suggestions": ["Acknowledge conservation counterevidence in the claim."],
            "text_errors": [],
        }
    )
    review = MainReviewBody.model_validate(
        {
            "review_scope": "main",
            "verdict": "REVISE",
            "rating": rating.model_dump(mode="json"),
            "issues": [],
            "required_changes": ["ADD_COUNTEREVIDENCE", "LOWER_CONFIDENCE", "NARROW_CLAIM"],
            "cited_evidence_ids": [],
            "explanation": _OBSERVED_EXPLANATION,
        }
    )
    assert review.explanation == _OBSERVED_EXPLANATION
    assert review.rating.rationale == _OBSERVED_RATIONALE


def test_main_explanation_still_has_a_hard_cap_above_observed_output() -> None:
    with pytest.raises(ValidationError, match="at most 2000 characters"):
        MainReviewBody.model_validate(
            {
                "review_scope": "main",
                "verdict": "APPROVE",
                "rating": {
                    "score": 4,
                    "rationale": "Bounded and supported.",
                    "suggestions": [],
                    "text_errors": [],
                },
                "issues": [],
                "required_changes": [],
                "cited_evidence_ids": [],
                "explanation": "x" * (CRITIC_EXPLANATION_MAX + 1),
            }
        )


def test_hypothesis_text_cap_matches_candidate_prose_limit() -> None:
    assert HYPOTHESIS_TEXT_MAX == CANDIDATE_PROSE_MAX == 800
    HypothesisBodyOutput.model_validate(
        {
            "statement": "s" * 800,
            "claim_modality": "association",
            "preferred_residues": {"39": ["C"]},
            "preference_strength_by_position": {"39": "exploratory"},
            "evidence_ids": ["E01"],
            "expected_outcome": "o" * 800,
            "falsification_criterion": "f" * 800,
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
        }
    )
