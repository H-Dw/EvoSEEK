---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:choose-display-platform-from-constraints
title: Choose the display platform from campaign constraints
language: en
knowledge_type: binding_display_selection
corpus_layer: binding
statement: "Choose phage, yeast, bacterial, ribosome, mRNA, or mammalian display from the required library size, folding environment, post-translational processing, quantitative readout, and recovery method instead of treating the platforms as interchangeable."
subject: binding library display platform
predicate: should_be_selected_by
object: library, folding, readout, and recovery constraints
polarity: support
claim_kind: operational_guideline
confidence: 0.92
applicability: {scope: display_based_binder_discovery_and_maturation, limitation: platform_specific_biases_require_soluble_validation}
citation_support:
  - support_id: binding:citation:choose-display-platform-review
    publication_id: doi:10.1039/b511782h
    support_type: background_support
    locator: abstract_comparison_of_molecular_display_platforms
    verified_against_source: true
  - support_id: binding:citation:choose-display-platform-ribosome
    publication_id: doi:10.1038/nmeth1003
    support_type: background_support
    locator: abstract_and_method_scope
    verified_against_source: true
selection_eligible: false
---
Choose phage, yeast, bacterial, ribosome, mRNA, or mammalian display from the required library size, folding environment, post-translational processing, quantitative readout, and recovery method instead of treating the platforms as interchangeable.
