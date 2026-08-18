---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:combine-validated-mutations-in-secondary-libraries
title: Combine validated mutations in secondary libraries
language: en
knowledge_type: binding_library_design
corpus_layer: binding
statement: "Build a secondary library from individually validated affinity or stability mutations across site groups, then remeasure the resulting combinations instead of assuming additive effects."
subject: secondary affinity-maturation library
predicate: should_combine
object: individually validated mutations with combination remeasurement
polarity: support
claim_kind: operational_guideline
confidence: 0.94
applicability: {scope: campaigns_with_multiple_enriched_site_or_loop_libraries, limitation: recombination_can_create_negative_epistasis_or_change_epitope}
citation_support:
  - support_id: binding:citation:combine-validated-mutations-phage
    publication_id: doi:10.1371/journal.pone.0129125
    support_type: direct_support
    locator: introduction_and_stepwise_CDR_optimization_workflow
    verified_against_source: true
  - support_id: binding:citation:combine-validated-mutations-liability-free
    publication_id: doi:10.1080/19420862.2022.2115200
    support_type: empirical_example
    locator: CDR_shuffling_and_final_clone_analysis
    verified_against_source: true
selection_eligible: false
---
Build a secondary library from individually validated affinity or stability mutations across site groups, then remeasure the resulting combinations instead of assuming additive effects.
