---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:preenrich-large-yeast-libraries
title: Pre-enrich large yeast libraries before fine sorting
language: en
knowledge_type: binding_display_selection
corpus_layer: binding
statement: "For a large yeast library with rare binders, perform a broad magnetic or bulk binding enrichment before quantitative FACS so that the cytometer is used for affinity resolution rather than initial library coverage."
subject: large yeast-displayed binding library
predicate: should_be_screened_by
object: broad pre-enrichment followed by quantitative FACS
polarity: support
claim_kind: operational_guideline
confidence: 0.89
applicability: {scope: yeast_libraries_larger_than_practical_single_pass_FACS_coverage, limitation: broad_preenrichment_can_lose_weak_or_presentation_sensitive_binders}
citation_support:
  - support_id: binding:citation:preenrich-large-yeast-libraries
    publication_id: doi:10.1038/nprot.2006.94
    support_type: method_basis
    locator: library_isolation_and_engineering_workflow
    verified_against_source: true
selection_eligible: false
---
For a large yeast library with rare binders, perform a broad magnetic or bulk binding enrichment before quantitative FACS so that the cytometer is used for affinity resolution rather than initial library coverage.
