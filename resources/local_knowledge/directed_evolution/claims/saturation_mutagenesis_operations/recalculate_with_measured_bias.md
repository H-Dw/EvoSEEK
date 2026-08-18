---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:recalculate-coverage-with-measured-codon-bias
title: Recalculate coverage from measured codon bias
language: en
knowledge_type: saturation_mutagenesis_operations
statement: "After pilot sequencing, recalculate saturation-library coverage with the observed wild-type and codon biases instead of the ideal degenerate-codon distribution."
subject: saturation-library coverage estimate
predicate: should_use
object: observed codon probabilities
polarity: support
claim_kind: operational_guideline
confidence: 0.93
applicability: {scope: saturation_libraries_with_pilot_sequence_data, limitation: shallow_pilots_give_uncertain_rare_variant_probabilities}
citation_support:
  - support_id: de:citation:recalculate-with-measured-bias
    publication_id: doi:10.1038/srep10654
    support_type: direct_support
    locator: quick_quality_control_results_and_wild_type_bias_analysis
    verified_against_source: true
selection_eligible: false
---
After pilot sequencing, recalculate saturation-library coverage with the observed wild-type and codon biases instead of the ideal degenerate-codon distribution.
