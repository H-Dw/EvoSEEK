---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:use-wild-type-inclusive-reduced-alphabets
title: Use wild-type-inclusive reduced alphabets
language: en
knowledge_type: binding_library_design
corpus_layer: binding
statement: "At each diversified interface position, include the wild-type residue and a justified reduced amino-acid set when full saturation would exceed the experimentally screenable library size."
subject: capacity-limited focused binding library
predicate: should_encode
object: wild type plus justified reduced amino-acid diversity
polarity: support
claim_kind: operational_guideline
confidence: 0.93
applicability: {scope: combinatorial_interface_or_CDR_libraries, limitation: reduced_alphabets_can_exclude_unexpected_beneficial_chemistries}
citation_support:
  - support_id: binding:citation:reduced-natural-diversity
    publication_id: doi:10.3389/fimmu.2017.00986
    support_type: direct_support
    locator: abstract_natural_diversity_degenerate_codon_design
    verified_against_source: true
  - support_id: binding:citation:reduced-small-perturbation
    publication_id: doi:10.1371/journal.pone.0129125
    support_type: empirical_example
    locator: abstract_and_small_perturbation_library_design
    verified_against_source: true
selection_eligible: false
---
At each diversified interface position, include the wild-type residue and a justified reduced amino-acid set when full saturation would exceed the experimentally screenable library size.
