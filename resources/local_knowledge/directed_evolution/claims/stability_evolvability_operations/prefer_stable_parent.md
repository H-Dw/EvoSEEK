---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:prefer-stable-parent-for-broad-mutagenesis
title: Prefer a stable parent for broad mutagenesis
language: en
knowledge_type: stability_evolvability_operations
statement: "When candidate starting parents have comparable target activity, prefer the parent with greater verified stability for broad mutagenesis because it is more likely to retain folding after additional substitutions."
subject: parent selection for broad mutagenesis
predicate: should_prefer
object: experimentally more stable functional parent
polarity: support
claim_kind: operational_guideline
confidence: 0.92
applicability: {scope: stability_limited_functional_evolution, limitation: stability_and_activity_must_be_measured_under_relevant_conditions}
citation_support:
  - support_id: de:citation:prefer-stable-parent
    publication_id: doi:10.1073/pnas.0510098103
    support_type: direct_support
    locator: abstract_and_p450_mutagenesis_results
    verified_against_source: true
selection_eligible: false
---
When candidate starting parents have comparable target activity, prefer the parent with greater verified stability for broad mutagenesis because it is more likely to retain folding after additional substitutions.
