---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:define-binding-objective
title: Define the binding quantity to optimize
language: en
knowledge_type: binding_campaign_definition
corpus_layer: binding
statement: "Before library construction, an affinity-maturation campaign should define which binding quantity it will optimize: equilibrium affinity, association rate, dissociation rate, or a functional binding threshold."
subject: binding-affinity campaign
predicate: should_define
object: KD, kon, koff, or functional binding threshold
polarity: support
claim_kind: operational_guideline
confidence: 0.93
applicability: {scope: protein_binder_affinity_maturation, limitation: functional_objectives_may_require_additional_cellular_or_biochemical_assays}
citation_support:
  - support_id: binding:citation:define-binding-objective-phage
    publication_id: doi:10.1016/0022-2836(92)90639-2
    support_type: method_basis
    locator: abstract_affinity_and_off_rate_selection
    verified_against_source: true
  - support_id: binding:citation:define-binding-objective-spr
    publication_id: doi:10.1146/annurev.biophys.26.1.541
    support_type: background_support
    locator: abstract_equilibrium_and_kinetic_parameters
    verified_against_source: true
selection_eligible: false
---
Before library construction, an affinity-maturation campaign should define which binding quantity it will optimize: equilibrium affinity, association rate, dissociation rate, or a functional binding threshold.
