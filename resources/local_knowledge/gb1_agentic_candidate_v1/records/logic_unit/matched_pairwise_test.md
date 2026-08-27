---
schema_version: scientific-logic-unit-runtime:v1
record_type: logic_unit
record_id: LU-AG-V1-MATCHED-COUPLING
logic_unit_id: LU-AG-V1-MATCHED-COUPLING
language: en
status: research_candidate
human_reviewed: false
selection_eligible: false
permission: explanation_only
knowledge_type: matched_pairwise_interaction
retrieval_text: A pairwise mechanism is testable only when the compared observations preserve the remaining background and the requested conservation or structural signal addresses the named pair; otherwise treat the apparent association as confounded.
scientific_quality:
  identity_verified: true
  span_verified: true
  entailment_status: verified
  source_credibility: 0.82
  independent_support_count: 1
  counterevidence_status: searched_found
  conflict_status: none
  uncertainty: 0.28
task_applicability:
  directness: direct
  context_match: 0.88
  candidate_discriminative_value: 0.75
  matched_dimensions: [pairwise_comparison, local_contact]
  unmatched_dimensions: [causal_epistasis]
  boundary_conditions: [The remaining sequence background must be controlled.]
boundary_conditions: [Do not infer causal epistasis from proximity or marginal association alone.]
counterclaims: [AC-AG-V1-STATIC-LIMIT]
falsifiers: [The same pairwise contrast is absent across background-matched observations.]
abstain_if: [No background-matched observations exist, or pairwise profile quality is degraded.]
question_leaf_id: [QL-LOCAL-COUPLING]
decision_slot: [interaction]
task_route: [mechanism_explanation]
feature_channel: [conservation, structure]
feature_focus: [conservation.pairwise_signal, conservation.profile_quality, structure.contact_geometry]
required_input: [matched_observation_pair, conservation.pairwise_signal, conservation.profile_quality, structure.contact_geometry]
expected_direction: [unknown]
evidence_role: [support]
stage: [any]
premise_claim_ids: [AC-AG-V1-STATIC-LIMIT]
---
A pairwise mechanism is testable only when the compared observations preserve the remaining background and the requested conservation or structural signal addresses the named pair; otherwise treat the apparent association as confounded.
