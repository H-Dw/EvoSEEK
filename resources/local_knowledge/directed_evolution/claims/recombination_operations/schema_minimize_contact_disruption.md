---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:schema-minimize-contact-disruption
title: Minimize contact disruption in divergent chimeras
language: en
knowledge_type: recombination_operations
statement: "For divergent homologs with a known structure, choose recombination crossover blocks that minimize disruption of residue-residue contacts before constructing chimeras."
subject: structure-guided recombination
predicate: should_minimize
object: disrupted residue contacts at crossover blocks
polarity: support
claim_kind: operational_guideline
confidence: 0.95
applicability: {scope: recombination_of_structurally_aligned_homologs, limitation: requires_reliable_alignment_and_structure}
citation_support:
  - support_id: de:citation:schema-minimize-contact-disruption
    publication_id: doi:10.1371/journal.pbio.0040112
    support_type: direct_support
    locator: abstract_introduction_and_schema_design
    verified_against_source: true
selection_eligible: false
---
For divergent homologs with a known structure, choose recombination crossover blocks that minimize disruption of residue-residue contacts before constructing chimeras.
