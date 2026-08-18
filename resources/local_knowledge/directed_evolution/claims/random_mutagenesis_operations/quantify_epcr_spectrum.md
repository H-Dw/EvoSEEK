---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:quantify-epcr-mutation-spectrum
title: Quantify the epPCR mutation spectrum
language: en
knowledge_type: random_mutagenesis_operations
statement: "From the epPCR pilot sequences, quantify transition and transversion bias, mutation positions, synonymous and nonsynonymous changes, and stop or frameshift frequency before choosing the production condition."
subject: epPCR pilot sequences
predicate: should_be_analyzed_for
object: mutation spectrum and truncation burden
polarity: support
claim_kind: operational_guideline
confidence: 0.90
applicability: {scope: error_prone_pcr_libraries, limitation: rare_events_need_deeper_sampling}
citation_support:
  - support_id: de:citation:quantify-epcr-spectrum
    publication_id: doi:10.1186/s12859-016-0996-7
    support_type: method_basis
    locator: implementation_and_results
    verified_against_source: true
selection_eligible: false
---
From the epPCR pilot sequences, quantify transition and transversion bias, mutation positions, synonymous and nonsynonymous changes, and stop or frameshift frequency before choosing the production condition.
