---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:choose-quantitative-screen-or-population-selection
title: Choose screen or selection from the required readout
language: en
knowledge_type: assay_engineering
statement: "Use a quantitative individual-variant screen when rank ordering matters, and use a population selection when library size makes individual measurement infeasible and the phenotype can be tightly coupled to survival or recovery."
subject: library analysis method
predicate: should_match
object: ranking need throughput and phenotype coupling
polarity: support
claim_kind: operational_guideline
confidence: 0.87
applicability: {scope: protein_library_analysis, limitation: selections_can_enrich_host_or_reporter_escape_mutations}
citation_support:
  - support_id: de:citation:choose-screen-or-selection
    publication_id: doi:10.3390/molecules26185599
    support_type: background_support
    locator: library_screening_methods
    verified_against_source: true
selection_eligible: false
---
Use a quantitative individual-variant screen when rank ordering matters, and use a population selection when library size makes individual measurement infeasible and the phenotype can be tightly coupled to survival or recovery.
