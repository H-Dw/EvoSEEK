---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:pooled-sequence-qc-before-saturation-screen
title: Perform pooled sequence QC before screening
language: en
knowledge_type: saturation_mutagenesis_operations
statement: "Perform pooled sequencing quality control on the saturation library before screening to verify that the intended bases were introduced at each randomized codon."
subject: saturation mutagenesis library
predicate: should_pass
object: pooled pre-screen sequencing quality control
polarity: support
claim_kind: operational_guideline
confidence: 0.94
applicability: {scope: degenerate_primer_saturation_libraries, limitation: pooled_traces_do_not_identify_linked_multi_site_haplotypes}
citation_support:
  - support_id: de:citation:pooled-sequence-qc
    publication_id: doi:10.1038/srep10654
    support_type: method_basis
    locator: quick_quality_control_methods_and_results
    verified_against_source: true
selection_eligible: false
---
Perform pooled sequencing quality control on the saturation library before screening to verify that the intended bases were introduced at each randomized codon.
