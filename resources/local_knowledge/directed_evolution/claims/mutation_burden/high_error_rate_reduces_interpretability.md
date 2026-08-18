---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:high-error-rate-reduces-interpretability
title: High mutation loads complicate functional library design
language: en
knowledge_type: mutation_burden
statement: "High-error-rate random mutagenesis changes the distribution of functional variants and makes individual mutation effects harder to attribute, so mutation load should be matched to screening capacity."
subject: high-error-rate mutagenesis
predicate: requires
object: mutation-load and screening-capacity control
polarity: support
claim_kind: evidence_informed_policy
confidence: 0.73
applicability: {scope: random_mutagenesis_libraries, limitation: no_universal_optimum}
citation_support:
  - support_id: de:citation:high-error-rate
    publication_id: doi:10.1016/j.jmb.2005.05.023
    support_type: background_support
    locator: article_scope
    verified_against_source: false
selection_eligible: false
---
High-error-rate random mutagenesis changes the distribution of functional variants and makes individual mutation effects harder to attribute, so mutation load should be matched to screening capacity.
