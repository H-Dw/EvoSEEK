---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:retain-bounded-exploration
title: Focused search should retain bounded exploration
language: en
knowledge_type: directed_evolution_strategy
statement: "A knowledge-guided protein campaign should retain a bounded exploration quota because focused priors reduce screening cost but can exclude improvements outside the current model."
subject: knowledge-guided protein campaign
predicate: should_retain
object: bounded exploration quota
polarity: support
claim_kind: evidence_informed_policy
confidence: 0.66
applicability: {scope: model_guided_directed_evolution, limitation: quota_requires_campaign_specific_tuning}
citation_support:
  - support_id: de:citation:navigating-sequence-space
    publication_id: doi:10.1039/c4cs00351a
    support_type: background_support
    locator: review_scope
    verified_against_source: false
selection_eligible: false
---
A knowledge-guided protein campaign should retain a bounded exploration quota because focused priors reduce screening cost but can exclude improvements outside the current model.
