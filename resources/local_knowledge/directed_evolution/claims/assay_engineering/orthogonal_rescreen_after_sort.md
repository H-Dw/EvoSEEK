---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:orthogonal-rescreen-after-enrichment
title: Verify enriched hits in an orthogonal assay
language: en
knowledge_type: assay_engineering
statement: "After an enrichment or droplet sort, re-isolate candidate genotypes and remeasure them in an independent lower-throughput assay before declaring improvement."
subject: enriched directed evolution hit
predicate: requires_confirmation_by
object: independent lower-throughput assay
polarity: support
claim_kind: operational_guideline
confidence: 0.93
applicability: {scope: enrichment_and_ultrahigh_throughput_screens, limitation: secondary_assay_must_preserve_the_target_phenotype}
citation_support:
  - support_id: de:citation:orthogonal-rescreen-after-sort
    publication_id: doi:10.1038/s41467-018-03492-6
    support_type: direct_support
    locator: results_library_2_and_96_well_verification
    verified_against_source: true
selection_eligible: false
---
After an enrichment or droplet sort, re-isolate candidate genotypes and remeasure them in an independent lower-throughput assay before declaring improvement.
