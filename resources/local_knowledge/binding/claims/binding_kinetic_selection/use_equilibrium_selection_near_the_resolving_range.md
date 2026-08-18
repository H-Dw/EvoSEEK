---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:use-equilibrium-selection-near-resolving-range
title: Use equilibrium selection near the resolving range
language: en
knowledge_type: binding_kinetic_selection
corpus_layer: binding
statement: "For equilibrium-affinity sorting, allow binding to equilibrate and set ligand concentration within the range that separates the parent from the desired higher-affinity population."
subject: equilibrium affinity selection
predicate: should_use
object: equilibrated ligand concentration in the resolving range
polarity: support
claim_kind: operational_guideline
confidence: 0.94
applicability: {scope: display_systems_with_quantitative_equilibrium_binding_readout, limitation: ultra_tight_binding_and_ligand_depletion_can_make_equilibrium_selection_impractical}
citation_support:
  - support_id: binding:citation:equilibrium-selection-titration
    publication_id: doi:10.7554/elife.23156
    support_type: method_basis
    locator: binding_titration_model_and_concentration_dependence
    verified_against_source: true
  - support_id: binding:citation:equilibrium-selection-yeast
    publication_id: doi:10.2174/138620708783744516
    support_type: background_support
    locator: affinity_maturation_section
    verified_against_source: true
selection_eligible: false
---
For equilibrium-affinity sorting, allow binding to equilibrate and set ligand concentration within the range that separates the parent from the desired higher-affinity population.
