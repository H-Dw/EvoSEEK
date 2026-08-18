---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:calibrate-dual-selection-stringency
title: Calibrate both arms of a dual selection
language: en
knowledge_type: specificity_engineering
statement: "Measure the parent or current intermediate against both reporters, then set positive and negative selection stringencies so neither arm eliminates all recoverable variants."
subject: dual selection system
predicate: should_calibrate
object: positive and negative stringency on the current parent
polarity: support
claim_kind: operational_guideline
confidence: 0.91
applicability: {scope: simultaneous_positive_negative_selection, limitation: reporter_response_can_change_with_genetic_background}
citation_support:
  - support_id: de:citation:calibrate-dual-selection-stringency
    publication_id: doi:10.1038/s41467-020-20650-x
    support_type: direct_support
    locator: abstract_and_counterselection_stringency_results
    verified_against_source: true
selection_eligible: false
---
Measure the parent or current intermediate against both reporters, then set positive and negative selection stringencies so neither arm eliminates all recoverable variants.
