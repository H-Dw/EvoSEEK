---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:calculate-saturation-coverage-before-construction
title: Calculate saturation-library coverage before construction
language: en
knowledge_type: saturation_mutagenesis_operations
statement: "Before construction, calculate expected amino-acid probabilities and the number of assayed clones needed for the chosen coverage target under the selected codon scheme."
subject: saturation mutagenesis library
predicate: requires_preconstruction_calculation_of
object: variant probabilities and assay count
polarity: support
claim_kind: operational_guideline
confidence: 0.94
applicability: {scope: degenerate_codon_saturation_libraries, limitation: calculation_requires_a_defined_coverage_metric}
citation_support:
  - support_id: de:citation:calculate-saturation-coverage
    publication_id: doi:10.1371/journal.pone.0068069
    support_type: method_basis
    locator: probability_and_library_size_methods
    verified_against_source: true
selection_eligible: false
---
Before construction, calculate expected amino-acid probabilities and the number of assayed clones needed for the chosen coverage target under the selected codon scheme.
