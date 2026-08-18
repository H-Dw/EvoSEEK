---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:set-screen-budget-before-library-design
title: Set screening capacity before library complexity
language: en
knowledge_type: campaign_definition
statement: "Set the maximum number of independent variants that can actually be assayed before choosing mutation sites, codon schemes, or combinatorial depth."
subject: directed evolution campaign planning
predicate: constrains_library_design_with
object: executable screening capacity
polarity: support
claim_kind: operational_guideline
confidence: 0.92
applicability: {scope: screening_based_directed_evolution, limitation: selection_based_campaigns_require_population_capacity_instead}
citation_support:
  - support_id: de:citation:set-screen-budget-first
    publication_id: doi:10.1371/journal.pone.0068069
    support_type: direct_support
    locator: introduction_and_library_size_methods
    verified_against_source: true
selection_eligible: false
---
Set the maximum number of independent variants that can actually be assayed before choosing mutation sites, codon schemes, or combinatorial depth.
