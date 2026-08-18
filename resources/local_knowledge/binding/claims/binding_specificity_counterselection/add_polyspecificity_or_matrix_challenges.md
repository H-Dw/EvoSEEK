---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:add-polyspecificity-or-matrix-challenges
title: Add polyspecificity or matrix challenges
language: en
knowledge_type: binding_specificity_counterselection
corpus_layer: binding
statement: "Include a defined polyspecificity reagent or a relevant complex non-target matrix during selection and clone testing to remove variants whose apparent affinity depends on nonspecific interactions."
subject: in vitro affinity-maturation campaign
predicate: should_include
object: polyspecificity or complex-matrix challenge
polarity: support
claim_kind: operational_guideline
confidence: 0.92
applicability: {scope: therapeutic_or_diagnostic_binders_exposed_to_complex_samples, limitation: the_challenge_reagent_must_match_the_intended_failure_mode}
citation_support:
  - support_id: binding:citation:polyspecificity-challenges
    publication_id: doi:10.1016/j.bej.2018.06.003
    support_type: background_support
    locator: specificity_tradeoff_and_negative_selection_section
    verified_against_source: true
selection_eligible: false
---
Include a defined polyspecificity reagent or a relevant complex non-target matrix during selection and clone testing to remove variants whose apparent affinity depends on nonspecific interactions.
