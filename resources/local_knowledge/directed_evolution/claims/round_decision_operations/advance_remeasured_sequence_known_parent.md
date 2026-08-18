---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:advance-remeasured-sequence-known-parent
title: Advance only a remeasured sequence-known parent
language: en
knowledge_type: round_decision_operations
statement: "Use an experimentally remeasured, sequence-identified variant as the next parent, and do not advance a candidate solely from a primary-screen score or model prediction."
subject: next-round parent decision
predicate: requires
object: remeasured phenotype and known sequence identity
polarity: support
claim_kind: operational_guideline
confidence: 0.95
applicability: {scope: iterative_directed_evolution, limitation: exact_arrayed_synthesis_can_replace_post_screen_sequencing}
citation_support:
  - support_id: de:citation:advance-remeasured-parent
    publication_id: doi:10.1038/s41467-018-03492-6
    support_type: empirical_example
    locator: results_96_well_verification_and_round_progression
    verified_against_source: true
selection_eligible: false
---
Use an experimentally remeasured, sequence-identified variant as the next parent, and do not advance a candidate solely from a primary-screen score or model prediction.
