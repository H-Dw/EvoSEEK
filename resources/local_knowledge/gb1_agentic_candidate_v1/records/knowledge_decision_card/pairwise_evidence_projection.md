---
schema_version: knowledge-decision-card-runtime:v1
record_type: knowledge_decision_card
record_id: DC-AG-V1-PAIRWISE-CHECK
decision_card_id: DC-AG-V1-PAIRWISE-CHECK
language: en
status: research_candidate
human_reviewed: false
selection_eligible: false
permission: explanation_only
knowledge_type: pairwise_evidence_projection
retrieval_text: Use this card to check whether a named pairwise hypothesis has both a background-matched assay contrast and usable pairwise profile or contact evidence. Request only the pairwise signal, profile quality, and contact geometry needed for that contrast; abstain when the comparison is confounded.
scientific_quality:
  logic_units: [LU-AG-V1-MATCHED-COUPLING]
task_applicability:
  directness: direct
  context_match: 0.88
  candidate_discriminative_value: 0.7
boundary_conditions: [Proximity and evolutionary co-occurrence do not establish causal assay epistasis.]
counterclaims: [AC-AG-V1-STATIC-LIMIT]
abstain_if: [No matched observation pair exists, or profile quality is degraded.]
required_inputs: [matched_observation_pair, conservation.pairwise_signal, conservation.profile_quality, structure.contact_geometry]
question_leaf_id: [QL-LOCAL-COUPLING]
decision_slot: [interaction]
task_route: [mechanism_explanation]
feature_channel: [conservation, structure]
feature_focus: [conservation.pairwise_signal, conservation.profile_quality, structure.contact_geometry]
required_input: [matched_observation_pair, conservation.pairwise_signal, conservation.profile_quality, structure.contact_geometry]
expected_direction: [unknown]
evidence_role: [boundary]
stage: [any]
logic_unit_ids: [LU-AG-V1-MATCHED-COUPLING]
---
Use this card to check whether a named pairwise hypothesis has both a background-matched assay contrast and usable pairwise profile or contact evidence. Request only the pairwise signal, profile quality, and contact geometry needed for that contrast; abstain when the comparison is confounded.
