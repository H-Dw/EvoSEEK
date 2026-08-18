---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:combinatorial-space-grows-exponentially
title: Multi-site sequence space grows exponentially
language: en
knowledge_type: mutation_burden
statement: "When each of n sites can take 20 amino-acid states, the complete combinatorial sequence space contains 20^n variants and rapidly exceeds typical experimental screening capacity."
subject: multi-site amino-acid library
predicate: has_space_size
object: 20^n variants
polarity: support
claim_kind: scientific_prior
confidence: 0.98
applicability: {scope: complete_twenty_amino_acid_site_randomization, limitation: encoding_and_exclusions_change_actual_size}
citation_support:
  - support_id: de:citation:four-site-landscape
    publication_id: doi:10.7554/elife.16965
    support_type: empirical_example
    locator: four-site_fitness_landscape
    verified_against_source: false
selection_eligible: false
---
When each of n sites can take 20 amino-acid states, the complete combinatorial sequence space contains 20^n variants and rapidly exceeds typical experimental screening capacity.
