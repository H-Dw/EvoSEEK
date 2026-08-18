---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:normalize-binding-by-display-expression
title: Normalize binding by display expression
language: en
knowledge_type: binding_affinity_measurement
corpus_layer: binding
statement: "Measure ligand binding and surface expression in separate fluorescence channels and gate on their joint distribution so that increased display is not mistaken for increased affinity."
subject: quantitative surface-display screen
predicate: should_measure
object: binding and expression in separate channels
polarity: support
claim_kind: operational_guideline
confidence: 0.95
applicability: {scope: yeast_or_cell_surface_display_with_expression_tag, limitation: expression_normalization_does_not_remove_avidity_or_ligand_depletion}
citation_support:
  - support_id: binding:citation:normalize-binding-by-display-expression
    publication_id: doi:10.1038/nprot.2006.94
    support_type: method_basis
    locator: flow_cytometry_binding_and_expression_protocol
    verified_against_source: true
  - support_id: binding:citation:normalize-binding-by-display-expression-review
    publication_id: doi:10.2174/138620708783744516
    support_type: background_support
    locator: affinity_maturation_and_quantitative_flow_cytometry_section
    verified_against_source: true
selection_eligible: false
---
Measure ligand binding and surface expression in separate fluorescence channels and gate on their joint distribution so that increased display is not mistaken for increased affinity.
