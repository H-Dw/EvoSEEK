---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:hydropathy-is-one-descriptor
title: Hydropathy is one substitution descriptor
language: en
knowledge_type: amino_acid_properties
statement: "A hydropathy scale can quantify one dimension of an amino-acid substitution, but it does not by itself determine whether the substitution improves protein fitness."
subject: amino-acid substitution
predicate: requires_context_beyond
object: hydropathy scale
polarity: support
claim_kind: scientific_prior
confidence: 0.78
applicability: {scope: canonical_amino_acid_substitutions, limitation: position_and_assay_context_required}
citation_support:
  - support_id: de:citation:hydropathy-kyte-doolittle
    publication_id: doi:10.1016/0022-2836(82)90515-0
    support_type: method_basis
    locator: title_and_method_scope
    verified_against_source: false
selection_eligible: false
---
A hydropathy scale can quantify one dimension of an amino-acid substitution, but it does not by itself determine whether the substitution improves protein fitness.
