---
schema_version: knowledge-decision-card-runtime:v1
record_type: knowledge_decision_card
record_id: DC-AG-V1-FOLD-BOUNDARY
decision_card_id: DC-AG-V1-FOLD-BOUNDARY
language: en
status: research_candidate
human_reviewed: false
selection_eligible: false
permission: explanation_only
knowledge_type: fold_boundary_projection
retrieval_text: Use this card only to explain whether a measured assay direction crosses a named fold-plausibility boundary. Request physicochemical special flags and structural interaction flags for the visible sample, retain all quality warnings, and abstain from ranking or residue preference.
scientific_quality:
  logic_units: [LU-AG-V1-MEASUREMENT-BOUNDARY]
task_applicability:
  directness: direct
  context_match: 0.9
  candidate_discriminative_value: 0.45
boundary_conditions: [A stability-plausible observation is not thereby a strong binder.]
counterclaims: [AC-AG-V1-STATIC-LIMIT]
abstain_if: [The sample lacks a revealed assay measurement, or either requested feature reports unavailable quality.]
required_inputs: [measured_assay_direction, physchem.special_flags, structure.interaction_flags]
question_leaf_id: [QL-FOLD-BINDING-SEPARATION]
decision_slot: [failure_mode]
task_route: [mechanism_explanation]
feature_channel: [physchem, structure]
feature_focus: [physchem.special_flags, structure.interaction_flags]
required_input: [measured_assay_direction, physchem.special_flags, structure.interaction_flags]
expected_direction: [unknown]
evidence_role: [boundary]
stage: [any]
logic_unit_ids: [LU-AG-V1-MEASUREMENT-BOUNDARY]
---
Use this card only to explain whether a measured assay direction crosses a named fold-plausibility boundary. Request physicochemical special flags and structural interaction flags for the visible sample, retain all quality warnings, and abstain from ranking or residue preference.
