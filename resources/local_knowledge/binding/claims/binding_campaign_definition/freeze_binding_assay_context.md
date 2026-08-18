---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:freeze-binding-assay-context
title: Freeze the binding assay context
language: en
knowledge_type: binding_campaign_definition
corpus_layer: binding
statement: "Record target format, valency, presentation, buffer, temperature, incubation time, and the required off-target panel before diversification so that round-to-round enrichment refers to the same binding objective."
subject: binding campaign definition
predicate: should_record
object: assay format and off-target context
polarity: support
claim_kind: operational_guideline
confidence: 0.90
applicability: {scope: iterative_binding_selection, limitation: deliberate_stringency_changes_must_be_versioned_instead_of_hidden}
citation_support:
  - support_id: binding:citation:freeze-binding-assay-context-display
    publication_id: doi:10.1039/b511782h
    support_type: background_support
    locator: abstract_display_platform_advantages_and_limitations
    verified_against_source: true
  - support_id: binding:citation:freeze-binding-assay-context-bli
    publication_id: doi:10.1016/j.xpro.2021.100836
    support_type: method_basis
    locator: before_you_begin_and_assay_design_controls
    verified_against_source: true
selection_eligible: false
---
Record target format, valency, presentation, buffer, temperature, incubation time, and the required off-target panel before diversification so that round-to-round enrichment refers to the same binding objective.
