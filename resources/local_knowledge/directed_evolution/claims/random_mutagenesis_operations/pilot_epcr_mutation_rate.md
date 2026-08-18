---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:pilot-epcr-mutation-rate
title: Measure the epPCR mutation rate before scaling
language: en
knowledge_type: random_mutagenesis_operations
statement: "Run a small error-prone PCR pilot under each candidate condition and sequence sampled clones to estimate mutations per gene before producing the full library."
subject: error-prone PCR library construction
predicate: requires_pilot_measurement_of
object: mutations per gene
polarity: support
claim_kind: operational_guideline
confidence: 0.94
applicability: {scope: error_prone_pcr_libraries, limitation: pilot_sample_size_limits_precision}
citation_support:
  - support_id: de:citation:pilot-epcr-cadwell
    publication_id: doi:10.1101/gr.2.1.28
    support_type: method_basis
    locator: abstract_and_method_scope
    verified_against_source: true
  - support_id: de:citation:pilot-epcr-mutanalyst
    publication_id: doi:10.1186/s12859-016-0996-7
    support_type: direct_support
    locator: background_and_results
    verified_against_source: true
selection_eligible: false
---
Run a small error-prone PCR pilot under each candidate condition and sequence sampled clones to estimate mutations per gene before producing the full library.
