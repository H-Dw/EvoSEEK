---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:prevent-ligand-depletion
title: Prevent ligand depletion during equilibrium selection
language: en
knowledge_type: binding_affinity_measurement
corpus_layer: binding
statement: "Before equilibrium sorting, estimate total displayed binding sites and use enough ligand and volume to keep free ligand approximately constant throughout the incubation."
subject: equilibrium display incubation
predicate: should_prevent
object: depletion of free ligand by the library
polarity: support
claim_kind: operational_guideline
confidence: 0.93
applicability: {scope: high_density_surface_display_at_limiting_ligand, limitation: very_tight_binders_may_require_impractically_large_volumes_and_kinetic_selection}
citation_support:
  - support_id: binding:citation:prevent-ligand-depletion
    publication_id: doi:10.1080/19420862.2022.2115200
    support_type: method_basis
    locator: equilibrium_selection_and_antigen_depletion_section
    verified_against_source: true
selection_eligible: false
---
Before equilibrium sorting, estimate total displayed binding sites and use enough ligand and volume to keep free ligand approximately constant throughout the incubation.
