---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:increase-chase-time-from-measured-off-rate
title: Increase chase time from the measured off-rate
language: en
knowledge_type: binding_kinetic_selection
corpus_layer: binding
statement: "Set the first kinetic-selection chase from the measured parent dissociation curve and lengthen later chases only when recovered diversity and control retention show that the new stringency remains resolvable."
subject: kinetic affinity-maturation schedule
predicate: should_adapt
object: chase duration from measured dissociation and recovery
polarity: support
claim_kind: operational_guideline
confidence: 0.92
applicability: {scope: iterative_slow_off_rate_selection, limitation: excessive_chase_can_select_assay_persistence_or_destroy_recoverable_complexes}
citation_support:
  - support_id: binding:citation:increase-chase-time-primary
    publication_id: doi:10.1073/pnas.170297297
    support_type: direct_support
    locator: abstract_and_iterative_kinetic_screening_results
    verified_against_source: true
  - support_id: binding:citation:increase-chase-time-screen
    publication_id: doi:10.1016/j.ab.2013.07.025
    support_type: method_basis
    locator: abstract_off_rate_screening_method
    verified_against_source: true
selection_eligible: false
---
Set the first kinetic-selection chase from the measured parent dissociation curve and lengthen later chases only when recovered diversity and control retention show that the new stringency remains resolvable.
