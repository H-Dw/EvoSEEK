---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:choose-reduced-codon-when-screening-is-expensive
title: Use reduced codons when screening is expensive
language: en
knowledge_type: saturation_mutagenesis_operations
statement: "When assay cost dominates primer cost, use a reduced-redundancy codon scheme that covers the required amino acids while removing stop codons and duplicated encodings."
subject: saturation mutagenesis codon scheme
predicate: should_minimize
object: stops and redundant encodings under screening cost
polarity: support
claim_kind: operational_guideline
confidence: 0.91
applicability: {scope: saturation_libraries_with_expensive_screens, limitation: reduced_schemes_can_require_more_primers_or_custom_synthesis}
citation_support:
  - support_id: de:citation:choose-reduced-codon-scheme
    publication_id: doi:10.1038/srep10654
    support_type: direct_support
    locator: cost_model_and_discussion
    verified_against_source: true
selection_eligible: false
---
When assay cost dominates primer cost, use a reduced-redundancy codon scheme that covers the required amino acids while removing stop codons and duplicated encodings.
