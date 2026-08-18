---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:sample-nonredundant-sequences-across-rounds
title: Sample nonredundant sequences across rounds
language: en
knowledge_type: binding_round_decision
corpus_layer: binding
statement: "Sequence intermediate and final pools, cluster identical or near-identical variants, and select nonredundant representatives for individual binding measurements instead of testing only the most abundant clone."
subject: enriched binding-library pools
predicate: should_be_sampled_by
object: cross-round sequencing and nonredundant clone selection
polarity: support
claim_kind: operational_guideline
confidence: 0.92
applicability: {scope: iterative_display_selection_with_sequence_recovery, limitation: sequence_diversity_does_not_guarantee_epitope_or_functional_diversity}
citation_support:
  - support_id: binding:citation:sample-nonredundant-phage
    publication_id: doi:10.1371/journal.pone.0129125
    support_type: direct_support
    locator: sequencing_and_enrichment_landscape_analysis
    verified_against_source: true
  - support_id: binding:citation:sample-nonredundant-ribosome
    publication_id: doi:10.1038/s41551-023-01093-3
    support_type: empirical_example
    locator: abstract_massively_parallel_sequencing_and_affinity_screening
    verified_against_source: true
selection_eligible: false
---
Sequence intermediate and final pools, cluster identical or near-identical variants, and select nonredundant representatives for individual binding measurements instead of testing only the most abundant clone.
