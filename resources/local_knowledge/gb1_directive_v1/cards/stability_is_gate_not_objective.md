---
schema_version: gb1-directive-card:v1
record_type: knowledge_decision_card
language: en
status: research_candidate
human_reviewed: false
selection_eligible: false
permission: explanation_only
benchmark_overlap: none
scientific_credibility: high
task_applicability: high_as_boundary
retrieval_similarity: runtime_only
card_id: DC-GB1-005
title: Use stability as a gate rather than the objective
inputs:
  positions: [39, 40, 41, 54]
  structural_context: folded_binding_domain
feature: separation_of_folding_and_binding
direction: constrain_then_diversify
boundaries:
  - Thermodynamic stability is not benchmark binding fitness.
  - A stable candidate can have weak Fc binding.
uncertainty: low_for_boundary_medium_for_ranking
knowledge_type: uncertainty_domain_shift
topics: [folding versus binding, proxy validity, confidence separation]
source_spans: [SS-003, SS-005]
logic_units: [LU-005]
---

## Use when

The reasoning trace ranks a candidate mainly because it appears more stable, more hydrophobic, or better packed, without a separate interface hypothesis.

## Decision rule

Use stability evidence to remove or downgrade likely folding failures; do not use it as the final binding objective.

## Candidate action

After applying the fold and packing gates, preserve interface diversity at position 40 and at least one alternative 39/54 packing class. Assign confidence separately for folding plausibility and binding plausibility.

## Matched comparison

Compare candidates with similar predicted fold risk but different interface chemistry, and candidates with similar interface chemistry but different fold risk. This two-axis design reveals which source of evidence drives the ranking.

## Abstain or downgrade when

Abstain from a binding conclusion when only thermodynamic stability evidence is available. Downgrade any single scalar that silently combines folding and affinity without calibration.

## Evidence basis

The comprehensive GB1 thermodynamic study states that fitness and stability are not interchangeable and that separating folding and binding improves interpretation. The rule protects against proxy collapse. It is falsified only if the runtime assay is explicitly defined and calibrated as a direct stability objective rather than Fc-binding fitness.
