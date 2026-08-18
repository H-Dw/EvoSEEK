---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:measure-polyreactivity-with-affinity
title: Measure polyreactivity with affinity
language: en
knowledge_type: binding_developability_validation
corpus_layer: binding
statement: "Measure nonspecific or polyreactive binding alongside target affinity because affinity-enhancing mutations can increase binding to unrelated molecules."
subject: affinity-enhancing mutation or clone
predicate: should_be_tested_for
object: target affinity and polyreactivity
polarity: support
claim_kind: operational_guideline
confidence: 0.95
applicability: {scope: therapeutic_diagnostic_and_complex_matrix_binders, limitation: selected_polyreactivity_assays_sample_only_a_subset_of_possible_nonspecific_interactions}
citation_support:
  - support_id: binding:citation:polyreactivity-tradeoff
    publication_id: doi:10.1016/j.bej.2018.06.003
    support_type: background_support
    locator: affinity_specificity_tradeoff_section
    verified_against_source: true
  - support_id: binding:citation:polyreactivity-primary
    publication_id: doi:10.1084/jem.182.3.743
    support_type: direct_support
    locator: abstract_increased_cognate_and_autoantigen_binding
    verified_against_source: true
selection_eligible: false
---
Measure nonspecific or polyreactive binding alongside target affinity because affinity-enhancing mutations can increase binding to unrelated molecules.
