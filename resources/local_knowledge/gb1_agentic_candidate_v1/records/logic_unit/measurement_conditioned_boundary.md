---
schema_version: scientific-logic-unit-runtime:v1
record_type: logic_unit
record_id: LU-AG-V1-MEASUREMENT-BOUNDARY
logic_unit_id: LU-AG-V1-MEASUREMENT-BOUNDARY
language: en
status: research_candidate
human_reviewed: false
selection_eligible: false
permission: explanation_only
knowledge_type: measurement_conditioned_boundary
retrieval_text: When current assay measurements and static fold descriptors disagree, preserve the measured direction and use physicochemical or structural flags only to state a fold-plausibility boundary; abstain from residue preference if the disagreement is not isolated by a matched comparison.
scientific_quality:
  identity_verified: true
  span_verified: true
  entailment_status: verified
  source_credibility: 0.88
  independent_support_count: 1
  counterevidence_status: searched_found
  conflict_status: none
  uncertainty: 0.2
task_applicability:
  directness: direct
  context_match: 0.9
  candidate_discriminative_value: 0.65
  matched_dimensions: [current_measurement, fold_boundary]
  unmatched_dimensions: [unmeasured_mutant_structure]
  boundary_conditions: [External evidence cannot override a revealed assay observation.]
boundary_conditions: [Static and physicochemical descriptors remain non-selecting explanations.]
counterclaims: [AC-AG-V1-STATIC-LIMIT]
falsifiers: [A preregistered matched comparison repeatedly shows the descriptor predicts the assay direction independently of current measurements.]
abstain_if: [No current measurement is visible, or the comparison changes multiple mechanisms.]
question_leaf_id: [QL-FOLD-BINDING-SEPARATION]
decision_slot: [failure_mode]
task_route: [mechanism_explanation]
feature_channel: [physchem, structure]
feature_focus: [physchem.special_flags, physchem.global_sequence_deltas, structure.interaction_flags]
required_input: [measured_assay_direction, physchem.special_flags, structure.interaction_flags]
expected_direction: [unknown]
evidence_role: [boundary]
stage: [any]
premise_claim_ids: [AC-AG-V1-FOLD-BINDING]
---
When current assay measurements and static fold descriptors disagree, preserve the measured direction and use physicochemical or structural flags only to state a fold-plausibility boundary; abstain from residue preference if the disagreement is not isolated by a matched comparison.
