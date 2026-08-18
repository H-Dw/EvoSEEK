---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:use-yeast-for-quantitative-two-channel-sorting
title: Use yeast for quantitative two-channel sorting
language: en
knowledge_type: binding_display_selection
corpus_layer: binding
statement: "Use yeast surface display when the campaign needs fluorescence-activated sorting of individual cells by both ligand binding and displayed protein expression."
subject: quantitative cell-based affinity screen
predicate: can_use
object: yeast surface display with two-channel FACS
polarity: support
claim_kind: operational_guideline
confidence: 0.94
applicability: {scope: proteins_compatible_with_yeast_surface_expression, limitation: yeast_processing_and_multivalent_display_can_differ_from_the_final_format}
citation_support:
  - support_id: binding:citation:use-yeast-two-channel-protocol
    publication_id: doi:10.1038/nprot.2006.94
    support_type: method_basis
    locator: yeast_display_library_sorting_protocol
    verified_against_source: true
  - support_id: binding:citation:use-yeast-two-channel-review
    publication_id: doi:10.2174/138620708783744516
    support_type: background_support
    locator: yeast_display_and_flow_cytometry_sections
    verified_against_source: true
selection_eligible: false
---
Use yeast surface display when the campaign needs fluorescence-activated sorting of individual cells by both ligand binding and displayed protein expression.
