---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:measure-round-zero-parent
title: Measure the parent before diversification
language: en
knowledge_type: campaign_definition
statement: "Before diversification, measure the starting protein in the same primary assay and operating conditions planned for library screening, and store that result as the round-zero reference."
subject: directed evolution campaign
predicate: begins_with
object: parent measurement in the production screening assay
polarity: support
claim_kind: operational_guideline
confidence: 0.88
applicability: {scope: protein_directed_evolution_campaigns, limitation: assay_must_be_available_for_the_parent}
citation_support:
  - support_id: de:citation:measure-round-zero-parent
    publication_id: doi:10.3390/molecules26185599
    support_type: background_support
    locator: directed_evolution_workflow_and_library_screening_sections
    verified_against_source: true
selection_eligible: false
---
Before diversification, measure the starting protein in the same primary assay and operating conditions planned for library screening, and store that result as the round-zero reference.
