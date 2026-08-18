---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: domain:claim:stable-slug
title: Concise English title
language: en
knowledge_type: replace_with_snake_case
statement: "One falsifiable English scientific statement."
subject: normalized subject
predicate: normalized_predicate
object: normalized object
polarity: support
claim_kind: scientific_prior
confidence: 0.70
applicability: {scope: define_scope, limitation: define_limitation}
citation_support:
  - support_id: domain:citation:stable-slug
    publication_id: doi:10.xxxx/example
    support_type: direct_support
    locator: abstract
    verified_against_source: false
selection_eligible: false
---
One falsifiable English scientific statement.
