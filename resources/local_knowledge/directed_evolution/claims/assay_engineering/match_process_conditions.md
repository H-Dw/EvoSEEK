---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:match-screen-to-process-conditions
title: Match the screen to operating conditions
language: en
knowledge_type: assay_engineering
statement: "Design the ranking assay to reproduce the operational substrate, temperature, pH, and stability challenge as closely as throughput allows, and confirm finalists under the full operating conditions."
subject: directed evolution ranking assay
predicate: should_reproduce
object: intended operating conditions
polarity: support
claim_kind: operational_guideline
confidence: 0.91
applicability: {scope: application_driven_enzyme_evolution, limitation: exact_conditions_may_require_lower_throughput_secondary_assay}
citation_support:
  - support_id: de:citation:match-process-conditions
    publication_id: doi:10.1038/nbt1286
    support_type: direct_support
    locator: optimization_strategy_and_assay_revalidation
    verified_against_source: true
selection_eligible: false
---
Design the ranking assay to reproduce the operational substrate, temperature, pH, and stability challenge as closely as throughput allows, and confirm finalists under the full operating conditions.
