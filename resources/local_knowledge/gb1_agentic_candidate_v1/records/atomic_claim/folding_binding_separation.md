---
schema_version: scientific-atomic-claim-runtime:v2
record_type: atomic_claim
record_id: AC-AG-V1-FOLD-BINDING
claim_id: AC-AG-V1-FOLD-BINDING
language: en
status: research_candidate
human_reviewed: false
selection_eligible: false
permission: explanation_only
knowledge_type: fold_binding_boundary
retrieval_text: Thermodynamic stability and assay binding fitness are distinct observables, so stability evidence can constrain fold plausibility but cannot by itself establish a favorable binding direction.
statement: Thermodynamic stability and assay binding fitness are distinct observables, so stability evidence can constrain fold plausibility but cannot by itself establish a favorable binding direction.
subject: stability evidence
predicate: constrains_without_determining
object: binding fitness
polarity: support
claim_kind: scientific_prior
scientific_quality:
  identity_verified: true
  span_verified: true
  entailment_status: verified
  source_credibility: 0.9
  independent_support_count: 1
  counterevidence_status: searched_found
  conflict_status: none
  uncertainty: 0.15
task_applicability:
  directness: direct
  context_match: 0.9
  candidate_discriminative_value: 0.4
  matched_dimensions: [folding, binding]
  unmatched_dimensions: [assay_condition_transfer]
  boundary_conditions: [No direct affinity measurement is supplied by a stability descriptor.]
boundary_conditions: [Do not convert stability or static packing into a binding rank.]
counterclaims: []
abstain_if: [The assay objective or condition transfer is unknown.]
question_leaf_id: [QL-FOLD-BINDING-SEPARATION]
decision_slot: [failure_mode]
task_route: [mechanism_explanation]
feature_channel: [physchem, structure]
required_input: [measured_assay_direction, structure.interaction_flags, physchem.special_flags]
expected_direction: [unknown]
evidence_role: [support]
stage: [any]
source_span_ids: [SS-003, SS-005]
---
Thermodynamic stability and assay binding fitness are distinct observables, so stability evidence can constrain fold plausibility but cannot by itself establish a favorable binding direction.
