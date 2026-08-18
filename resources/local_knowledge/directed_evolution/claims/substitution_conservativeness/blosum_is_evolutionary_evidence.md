---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:blosum-is-evolutionary-evidence
title: BLOSUM scores encode observed evolutionary substitutions
language: en
knowledge_type: substitution_conservativeness
statement: "BLOSUM scores summarize substitutions observed in conserved protein blocks and therefore provide evolutionary evidence, not a direct assay-specific fitness value."
subject: BLOSUM score
predicate: represents
object: observed evolutionary substitution evidence
polarity: support
claim_kind: scientific_prior
confidence: 0.84
applicability: {scope: protein_substitution_prior, limitation: not_assay_specific}
citation_support:
  - support_id: de:citation:blosum-blocks
    publication_id: doi:10.1073/pnas.89.22.10915
    support_type: direct_support
    locator: title_and_method
    verified_against_source: false
selection_eligible: false
---
BLOSUM scores summarize substitutions observed in conserved protein blocks and therefore provide evolutionary evidence, not a direct assay-specific fitness value.
