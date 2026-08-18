---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:compare-chimeras-to-all-parents
title: Compare chimeras with every recombination parent
language: en
knowledge_type: recombination_operations
statement: "Assay all recombination parents alongside chimeras so any gain is assigned relative to the best parent rather than to an arbitrary baseline."
subject: recombinant chimera assay
predicate: should_include
object: all parental controls
polarity: support
claim_kind: operational_guideline
confidence: 0.88
applicability: {scope: multi_parent_recombination_libraries, limitation: parental_expression_differences_require_normalization}
citation_support:
  - support_id: de:citation:compare-chimeras-to-parents
    publication_id: doi:10.1371/journal.pbio.0040112
    support_type: empirical_example
    locator: results_parent_and_chimera_comparisons
    verified_against_source: true
selection_eligible: false
---
Assay all recombination parents alongside chimeras so any gain is assigned relative to the best parent rather than to an arbitrary baseline.
