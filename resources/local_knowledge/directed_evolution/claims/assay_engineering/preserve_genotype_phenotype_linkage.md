---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:preserve-recoverable-genotype-phenotype-linkage
title: Preserve recoverable genotype-phenotype linkage
language: en
knowledge_type: assay_engineering
statement: "Each screening compartment must retain a recoverable link between the measured protein phenotype and the DNA sequence that encoded it."
subject: directed evolution screening compartment
predicate: must_retain
object: recoverable genotype-phenotype linkage
polarity: support
claim_kind: operational_guideline
confidence: 0.96
applicability: {scope: all_library_screens_and_selections, limitation: pooled_assays_need_barcode_or_physical_linkage}
citation_support:
  - support_id: de:citation:preserve-genotype-phenotype-linkage
    publication_id: doi:10.3390/molecules26185599
    support_type: background_support
    locator: library_screening_methods
    verified_against_source: true
selection_eligible: false
---
Each screening compartment must retain a recoverable link between the measured protein phenotype and the DNA sequence that encoded it.
