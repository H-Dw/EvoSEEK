---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:pool-independent-low-cycle-epcrs
title: Pool independent epPCR reactions to reduce jackpot bias
language: en
knowledge_type: random_mutagenesis_operations
statement: "When amplification jackpot bias matters, combine products from several independent low-cycle epPCR reactions instead of relying on one highly amplified reaction."
subject: epPCR production library
predicate: should_pool
object: independent low-cycle amplification reactions
polarity: support
claim_kind: operational_guideline
confidence: 0.85
applicability: {scope: amplification_based_random_mutagenesis, limitation: pooling_does_not_remove_polymerase_substitution_bias}
citation_support:
  - support_id: de:citation:pool-independent-epcrs
    publication_id: doi:10.1039/c4cs00351a
    support_type: background_support
    locator: random_mutagenesis_bias_section
    verified_against_source: true
selection_eligible: false
---
When amplification jackpot bias matters, combine products from several independent low-cycle epPCR reactions instead of relying on one highly amplified reaction.
