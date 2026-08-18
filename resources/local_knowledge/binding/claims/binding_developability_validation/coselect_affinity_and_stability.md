---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:coselect-affinity-and-stability
title: Co-select affinity and stability
language: en
knowledge_type: binding_developability_validation
corpus_layer: binding
statement: "Measure expression or folding stability during affinity maturation and retain compensatory mutations when affinity-enhancing substitutions destabilize the binding domain."
subject: affinity-matured binding domain
predicate: should_be_coselected_for
object: affinity and folding stability
polarity: support
claim_kind: operational_guideline
confidence: 0.95
applicability: {scope: binding_domains_where_destabilization_reduces_expression_or_soluble_yield, limitation: display_expression_is_only_a_proxy_for_thermodynamic_stability}
citation_support:
  - support_id: binding:citation:coselect-affinity-stability
    publication_id: doi:10.1038/srep45259
    support_type: direct_support
    locator: abstract_and_compensatory_mutation_results
    verified_against_source: true
selection_eligible: false
---
Measure expression or folding stability during affinity maturation and retain compensatory mutations when affinity-enhancing substitutions destabilize the binding domain.
