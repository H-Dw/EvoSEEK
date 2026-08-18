---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:pilot-z-prime-before-screen
title: Validate the screening window before library screening
language: en
knowledge_type: assay_engineering
statement: "Before screening a directed-evolution library with a quantitative assay, run replicated positive and negative controls and calculate Z-prime to confirm that the assay window and variance can separate hits from background."
subject: high-throughput protein assay
predicate: should_be_validated_by
object: replicated controls and Z-prime
polarity: support
claim_kind: operational_guideline
confidence: 0.94
applicability: {scope: quantitative_high_throughput_screens, limitation: threshold_must_be_interpreted_with_assay_context}
citation_support:
  - support_id: de:citation:pilot-z-prime
    publication_id: doi:10.1177/108705719900400206
    support_type: method_basis
    locator: abstract_and_z_factor_definition
    verified_against_source: true
selection_eligible: false
---
Before screening a directed-evolution library with a quantitative assay, run replicated positive and negative controls and calculate Z-prime to confirm that the assay window and variance can separate hits from background.
