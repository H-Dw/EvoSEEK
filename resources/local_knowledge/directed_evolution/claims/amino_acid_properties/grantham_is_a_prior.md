---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:grantham-is-a-prior
title: Grantham distance is a context-free prior
language: en
knowledge_type: amino_acid_properties
statement: "Grantham distance summarizes composition, polarity, and molecular-volume differences and should be treated as a context-free substitution prior rather than a site-specific fitness measurement."
subject: Grantham distance
predicate: provides
object: context-free substitution prior
polarity: support
claim_kind: scientific_prior
confidence: 0.82
applicability: {scope: canonical_amino_acid_substitutions, limitation: not_site_specific}
citation_support:
  - support_id: de:citation:grantham-prior
    publication_id: doi:10.1126/science.185.4154.862
    support_type: direct_support
    locator: amino_acid_difference_formula
    verified_against_source: false
selection_eligible: false
---
Grantham distance summarizes composition, polarity, and molecular-volume differences and should be treated as a context-free substitution prior rather than a site-specific fitness measurement.
