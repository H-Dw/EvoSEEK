---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:default-high-order-penalty
title: Unvalidated high-order variants need a calibrated penalty
language: en
knowledge_type: mutation_burden
statement: "A campaign may assign an empirically calibrated penalty to candidates containing at least four unvalidated amino-acid edits because attribution and interaction uncertainty increase with mutation count."
subject: candidate with at least four unvalidated edits
predicate: may_receive
object: calibrated mutation-burden penalty
polarity: support
claim_kind: evidence_informed_policy
confidence: 0.62
applicability: {scope: point_mutation_campaigns, limitation: only_after_campaign_specific_calibration}
citation_support:
  - support_id: de:citation:mutation-load-policy
    publication_id: doi:10.1016/j.jmb.2005.05.023
    support_type: background_support
    locator: article_scope
    verified_against_source: false
selection_eligible: true
---
A campaign may assign an empirically calibrated penalty to candidates containing at least four unvalidated amino-acid edits because attribution and interaction uncertainty increase with mutation count.
