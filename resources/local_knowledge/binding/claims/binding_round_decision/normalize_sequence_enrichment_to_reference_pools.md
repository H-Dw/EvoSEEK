---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:normalize-sequence-enrichment-to-reference-pools
title: Normalize sequence enrichment to reference pools
language: en
knowledge_type: binding_round_decision
corpus_layer: binding
statement: "Compute sequence enrichment relative to the unselected input and, when available, an expression-selected reference pool so that clone abundance is not interpreted as binding improvement by itself."
subject: display selection sequence counts
predicate: should_be_normalized_to
object: input and expression-selected reference pools
polarity: support
claim_kind: operational_guideline
confidence: 0.93
applicability: {scope: display_selection_with_NGS_readout, limitation: normalized_enrichment_remains_a_relative_assay_measure_not_an_absolute_KD}
citation_support:
  - support_id: binding:citation:normalize-enrichment-phage
    publication_id: doi:10.1371/journal.pone.0129125
    support_type: direct_support
    locator: deep_sequencing_of_unselected_and_selected_libraries
    verified_against_source: true
  - support_id: binding:citation:normalize-enrichment-multiplex
    publication_id: doi:10.1039/c9me00118b
    support_type: direct_support
    locator: methods_reference_pool_normalization
    verified_against_source: true
selection_eligible: false
---
Compute sequence enrichment relative to the unselected input and, when available, an expression-selected reference pool so that clone abundance is not interpreted as binding improvement by itself.
