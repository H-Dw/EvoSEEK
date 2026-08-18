---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:triage-chimeras-for-fold-before-function
title: Triage recombinant chimeras for folding
language: en
knowledge_type: recombination_operations
statement: "When a fold or cofactor-incorporation readout exists, triage recombinant chimeras on that readout before the functional assay to remove structurally disrupted variants cheaply."
subject: recombinant chimera library
predicate: should_be_triaged_by
object: fold or cofactor-incorporation readout
polarity: support
claim_kind: operational_guideline
confidence: 0.89
applicability: {scope: chimeric_proteins_with_fold_or_cofactor_reporters, limitation: correct_fold_does_not_guarantee_target_activity}
citation_support:
  - support_id: de:citation:triage-chimeras-for-fold
    publication_id: doi:10.1371/journal.pbio.0040112
    support_type: empirical_example
    locator: abstract_and_heme_incorporation_results
    verified_against_source: true
selection_eligible: false
---
When a fold or cofactor-incorporation readout exists, triage recombinant chimeras on that readout before the functional assay to remove structurally disrupted variants cheaply.
