---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:remeasure-components-and-combinations
title: Remeasure components and combinations
language: en
knowledge_type: binding_round_decision
corpus_layer: binding
statement: "When an improved clone carries several mutations, construct and measure informative single or subset variants alongside the full combination before attributing the affinity gain or reusing the mutations."
subject: multi-mutation affinity-improved clone
predicate: should_be_decomposed_into
object: measured component and combination variants
polarity: support
claim_kind: operational_guideline
confidence: 0.93
applicability: {scope: multi_mutation_clones_used_for_mechanism_or_recombination, limitation: exhaustive_decomposition_can_exceed_assay_capacity}
citation_support:
  - support_id: binding:citation:remeasure-components-phage
    publication_id: doi:10.1371/journal.pone.0129125
    support_type: direct_support
    locator: stepwise_mutation_interrogation_and_recombination_workflow
    verified_against_source: true
  - support_id: binding:citation:remeasure-components-shuffling
    publication_id: doi:10.1080/19420862.2022.2115200
    support_type: empirical_example
    locator: CDR_shuffling_and_clone_analysis
    verified_against_source: true
selection_eligible: false
---
When an improved clone carries several mutations, construct and measure informative single or subset variants alongside the full combination before attributing the affinity gain or reusing the mutations.
