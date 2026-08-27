---
schema_version: scientific-atomic-claim-runtime:v2
record_type: atomic_claim
record_id: AC-AG-V1-STATIC-LIMIT
claim_id: AC-AG-V1-STATIC-LIMIT
language: en
status: research_candidate
human_reviewed: false
selection_eligible: false
permission: explanation_only
knowledge_type: static_structure_boundary
retrieval_text: Wild-type contact proximity identifies a possible interaction axis, but an unrelaxed static structure does not determine the sign or magnitude of a mutant pair effect.
statement: Wild-type contact proximity identifies a possible interaction axis, but an unrelaxed static structure does not determine the sign or magnitude of a mutant pair effect.
subject: wild-type contact proximity
predicate: identifies_without_quantifying
object: mutant pair effect
polarity: support
claim_kind: scientific_prior
scientific_quality:
  identity_verified: true
  span_verified: true
  entailment_status: verified
  source_credibility: 0.85
  independent_support_count: 1
  counterevidence_status: searched_found
  conflict_status: none
  uncertainty: 0.25
task_applicability:
  directness: direct
  context_match: 0.85
  candidate_discriminative_value: 0.55
  matched_dimensions: [local_geometry]
  unmatched_dimensions: [mutant_relaxation, assay_effect]
  boundary_conditions: [Mutant side-chain relaxation and measured epistasis are unavailable.]
boundary_conditions: [Require measured or matched-comparison evidence before assigning direction.]
counterclaims: []
abstain_if: [The requested conclusion requires mutant affinity or causal epistasis.]
question_leaf_id: [QL-LOCAL-COUPLING]
decision_slot: [interaction]
task_route: [mechanism_explanation]
feature_channel: [structure]
required_input: [structure.contact_geometry, structure.interface_contacts, measured_pairwise_comparison]
expected_direction: [unknown]
evidence_role: [boundary]
stage: [any]
source_span_ids: [SS-001, SS-006]
---
Wild-type contact proximity identifies a possible interaction axis, but an unrelaxed static structure does not determine the sign or magnitude of a mutant pair effect.
