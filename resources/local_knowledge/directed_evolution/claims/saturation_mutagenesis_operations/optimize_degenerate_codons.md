---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:optimize-degenerate-codons-under-library-cap
title: Optimize degenerate codons under a library cap
language: en
knowledge_type: saturation_mutagenesis_operations
statement: "For a focused multi-site library, optimize degenerate codons against an explicit maximum DNA library size and a desired amino-acid set at each position."
subject: focused multi-site library
predicate: should_optimize
object: degenerate codons under a size cap and amino-acid targets
polarity: support
claim_kind: operational_guideline
confidence: 0.93
applicability: {scope: focused_combinatorial_degenerate_codon_libraries, limitation: input_amino_acid_weights_require_external_evidence}
citation_support:
  - support_id: de:citation:optimize-degenerate-codons
    publication_id: doi:10.1093/nar/gku1323
    support_type: method_basis
    locator: abstract_methods_and_discussion
    verified_against_source: true
selection_eligible: false
---
For a focused multi-site library, optimize degenerate codons against an explicit maximum DNA library size and a desired amino-acid set at each position.
