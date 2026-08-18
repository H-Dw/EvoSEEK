---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:use-droplets-when-plate-capacity-is-insufficient
title: Use droplets when plate capacity is insufficient
language: en
knowledge_type: droplet_screening_operations
statement: "Use droplet screening when the assay can be compartmentalized and the required library coverage exceeds practical microplate throughput."
subject: large directed evolution library
predicate: can_be_screened_with
object: compartmentalized droplet assay
polarity: support
claim_kind: operational_guideline
confidence: 0.92
applicability: {scope: optical_or_sortable_droplet_assays, limitation: assay_chemistry_and_recovery_must_be_microfluidic_compatible}
citation_support:
  - support_id: de:citation:use-droplets-for-large-libraries
    publication_id: doi:10.1073/pnas.1606927113
    support_type: direct_support
    locator: significance_abstract_and_discussion
    verified_against_source: true
selection_eligible: false
---
Use droplet screening when the assay can be compartmentalized and the required library coverage exceeds practical microplate throughput.
