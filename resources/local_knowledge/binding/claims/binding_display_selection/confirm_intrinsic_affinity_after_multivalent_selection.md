---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:confirm-intrinsic-affinity-after-multivalent-selection
title: Confirm intrinsic affinity after multivalent selection
language: en
knowledge_type: binding_display_selection
corpus_layer: binding
statement: "After a multivalent phage, yeast, or cell-panning selection, confirm the recovered clone in a monovalent soluble format before assigning an intrinsic affinity improvement."
subject: clone recovered by multivalent selection
predicate: should_be_confirmed_in
object: monovalent soluble binding assay
polarity: support
claim_kind: operational_guideline
confidence: 0.96
applicability: {scope: multivalent_display_or_cell_panning, limitation: avidity_can_remain_the_relevant_metric_for_some_native_multivalent_applications}
citation_support:
  - support_id: binding:citation:confirm-after-cell-panning
    publication_id: doi:10.1007/978-1-4939-9853-1_17
    support_type: method_basis
    locator: avidity_and_titratable_avidity_reduction_sections
    verified_against_source: true
  - support_id: binding:citation:confirm-monovalent-example
    publication_id: doi:10.1073/pnas.170297297
    support_type: empirical_example
    locator: abstract_monovalent_affinity
    verified_against_source: true
selection_eligible: false
---
After a multivalent phage, yeast, or cell-panning selection, confirm the recovered clone in a monovalent soluble format before assigning an intrinsic affinity improvement.
