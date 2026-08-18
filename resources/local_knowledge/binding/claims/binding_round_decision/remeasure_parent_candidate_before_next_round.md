---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:remeasure-parent-candidate-before-next-round
title: Remeasure a parent candidate before the next round
language: en
knowledge_type: binding_round_decision
corpus_layer: binding
statement: "Use a sequence-confirmed clone as the next-round parent only after soluble remeasurement shows an affinity or kinetic improvement over the parent under the same assay conditions."
subject: next-round binding parent candidate
predicate: should_require
object: sequence confirmation and matched soluble remeasurement
polarity: support
claim_kind: operational_guideline
confidence: 0.96
applicability: {scope: iterative_affinity_maturation_with_purifiable_clones, limitation: matched_binding_improvement_may_still_fail_a_separate_functional_assay}
citation_support:
  - support_id: binding:citation:remeasure-parent-bli
    publication_id: doi:10.1016/j.xpro.2021.100836
    support_type: method_basis
    locator: affinity_measurement_protocol
    verified_against_source: true
  - support_id: binding:citation:remeasure-parent-display
    publication_id: doi:10.1038/nprot.2006.94
    support_type: method_basis
    locator: clone_isolation_and_affinity_characterization_workflow
    verified_against_source: true
selection_eligible: false
---
Use a sequence-confirmed clone as the next-round parent only after soluble remeasurement shows an affinity or kinetic improvement over the parent under the same assay conditions.
