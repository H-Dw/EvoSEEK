---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:counterselect-predeclared-off-target-panel
title: Counterselect a predeclared off-target panel
language: en
knowledge_type: binding_specificity_counterselection
corpus_layer: binding
statement: "During affinity maturation, apply negative selection or explicit screening against predeclared homologs, matrix components, tags, and application-relevant off-targets rather than optimizing target binding alone."
subject: affinity-maturation selection
predicate: should_include
object: negative selection against a predeclared off-target panel
polarity: support
claim_kind: operational_guideline
confidence: 0.95
applicability: {scope: binders_requiring_molecular_specificity, limitation: a_finite_panel_cannot_exclude_all_unknown_off_targets}
citation_support:
  - support_id: binding:citation:counterselect-off-target-panel-parallel
    publication_id: doi:10.1039/c9me00118b
    support_type: direct_support
    locator: abstract_and_on_off_target_selection_results
    verified_against_source: true
  - support_id: binding:citation:counterselect-off-target-panel-computational
    publication_id: doi:10.1016/j.crmeth.2022.100254
    support_type: direct_support
    locator: abstract_and_cross_target_validation
    verified_against_source: true
selection_eligible: false
---
During affinity maturation, apply negative selection or explicit screening against predeclared homologs, matrix components, tags, and application-relevant off-targets rather than optimizing target binding alone.
