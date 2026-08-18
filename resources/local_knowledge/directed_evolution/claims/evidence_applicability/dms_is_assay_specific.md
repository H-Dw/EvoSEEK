---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:dms-is-assay-specific
title: Deep mutational scanning evidence is assay-specific
language: en
knowledge_type: evidence_applicability
statement: "Deep mutational scanning provides large-scale empirical variant measurements, but the resulting effects remain conditional on the assayed construct, phenotype, and experimental conditions."
subject: deep mutational scanning evidence
predicate: is_conditional_on
object: assay construct phenotype and conditions
polarity: support
claim_kind: scientific_prior
confidence: 0.86
applicability: {scope: external_variant_effect_data, limitation: transfer_requires_assay_alignment}
citation_support:
  - support_id: de:citation:dms-assay-scope
    publication_id: doi:10.1038/nmeth.3027
    support_type: background_support
    locator: review_scope
    verified_against_source: false
selection_eligible: false
---
Deep mutational scanning provides large-scale empirical variant measurements, but the resulting effects remain conditional on the assayed construct, phenotype, and experimental conditions.
