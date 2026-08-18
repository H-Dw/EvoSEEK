---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:fit-referenced-sensorgrams
title: Fit referenced sensorgrams over a concentration series
language: en
knowledge_type: binding_affinity_measurement
corpus_layer: binding
statement: "Collect association and dissociation sensorgrams at several analyte concentrations with a blank or reference channel and fit only models that explain the complete concentration series."
subject: SPR or BLI kinetic assay
predicate: should_fit
object: referenced multi-concentration sensorgrams
polarity: support
claim_kind: operational_guideline
confidence: 0.92
applicability: {scope: label_free_kinetic_measurement, limitation: heterogeneous_or_multivalent_interactions_may_not_follow_a_one_to_one_model}
citation_support:
  - support_id: binding:citation:fit-referenced-sensorgrams
    publication_id: doi:10.1016/j.xpro.2021.100836
    support_type: method_basis
    locator: figures_2_to_4_plate_layout_controls_and_sensorgrams
    verified_against_source: true
  - support_id: binding:citation:fit-referenced-sensorgrams-design
    publication_id: doi:10.1016/j.ab.2017.08.002
    support_type: method_basis
    locator: abstract_binding_kinetic_assay_design
    verified_against_source: true
selection_eligible: false
---
Collect association and dissociation sensorgrams at several analyte concentrations with a blank or reference channel and fit only models that explain the complete concentration series.
