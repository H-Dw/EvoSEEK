---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:validate-soluble-clones-by-label-free-kinetics
title: Validate soluble clones by label-free kinetics
language: en
knowledge_type: binding_affinity_measurement
corpus_layer: binding
statement: "Express selected clones as soluble proteins and remeasure kon, koff, and KD by SPR or BLI before declaring a display-enriched clone affinity improved."
subject: display-enriched binding clone
predicate: should_be_remeasured_by
object: soluble SPR or BLI kinetics
polarity: support
claim_kind: operational_guideline
confidence: 0.95
applicability: {scope: affinity_matured_proteins_that_can_be_purified, limitation: immobilization_and_mass_transport_artifacts_still_require_controls}
citation_support:
  - support_id: binding:citation:validate-soluble-clones-octet
    publication_id: doi:10.1016/j.ab.2008.03.035
    support_type: method_basis
    locator: abstract_parallel_label_free_kinetic_and_affinity_measurement
    verified_against_source: true
  - support_id: binding:citation:validate-soluble-clones-bli
    publication_id: doi:10.1016/j.xpro.2021.100836
    support_type: method_basis
    locator: affinity_measurement_workflow
    verified_against_source: true
selection_eligible: false
---
Express selected clones as soluble proteins and remeasure kon, koff, and KD by SPR or BLI before declaring a display-enriched clone affinity improved.
