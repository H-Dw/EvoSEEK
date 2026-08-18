---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:advance-on-a-multiassay-profile
title: Advance clones on a multi-assay profile
language: en
knowledge_type: binding_developability_validation
corpus_layer: binding
statement: "Advance an affinity-matured clone only after soluble affinity, kinetic, specificity, expression, monomer fraction or aggregation, and stability measurements are reviewed together against the campaign thresholds."
subject: affinity-matured lead clone
predicate: should_be_advanced_by
object: joint binding and developability profile
polarity: support
claim_kind: operational_guideline
confidence: 0.94
applicability: {scope: leads_intended_for_reagent_diagnostic_or_therapeutic_development, limitation: required_assays_and_thresholds_depend_on_the_final_product_context}
citation_support:
  - support_id: binding:citation:advance-multiassay-mammalian-display
    publication_id: doi:10.1080/19420862.2020.1829335
    support_type: direct_support
    locator: abstract_and_binding_polyreactivity_aggregation_results
    verified_against_source: true
  - support_id: binding:citation:advance-multiassay-tradeoffs
    publication_id: doi:10.1016/j.bej.2018.06.003
    support_type: background_support
    locator: affinity_specificity_stability_solubility_tradeoff_review
    verified_against_source: true
selection_eligible: false
---
Advance an affinity-matured clone only after soluble affinity, kinetic, specificity, expression, monomer fraction or aggregation, and stability measurements are reviewed together against the campaign thresholds.
