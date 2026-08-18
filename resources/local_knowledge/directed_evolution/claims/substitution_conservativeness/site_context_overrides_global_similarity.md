---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:site-context-overrides-global-similarity
title: Site context can override global substitution similarity
language: en
knowledge_type: substitution_conservativeness
statement: "A globally conservative amino-acid substitution can still disrupt function at a constrained site, so site-specific evolutionary and structural context must override a context-free similarity label."
subject: site-specific context
predicate: can_override
object: context-free substitution similarity
polarity: support
claim_kind: scientific_prior
confidence: 0.78
applicability: {scope: missense_substitution_interpretation, limitation: requires_homolog_quality}
citation_support:
  - support_id: de:citation:sift-site-specific
    publication_id: doi:10.1093/nar/gkg509
    support_type: method_basis
    locator: method_scope
    verified_against_source: false
selection_eligible: false
---
A globally conservative amino-acid substitution can still disrupt function at a constrained site, so site-specific evolutionary and structural context must override a context-free similarity label.
