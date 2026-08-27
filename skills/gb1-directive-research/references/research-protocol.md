# Research protocol

## Frozen question

Which source-grounded, label-independent structural and biophysical rules can help a downstream reasoning agent prioritize GB1 four-position mutation classes while preserving diversity and knowing when to abstain?

## Source order

1. Official structure records from RCSB/wwPDB or partner archives.
2. Primary peer-reviewed structure, biophysics, stability, and binding papers.
3. Publisher or DOI metadata for verification.
4. Semantic Scholar for discovery and citation navigation.
5. PubMed only as a metadata fallback.

Do not use OpenAlex. Do not substitute review prose for a primary source when the primary source is accessible.

## Query families

- GB1 fold, hydrophobic core, beta sheet, helix, structural neighbors.
- Protein G B1 Fc complex, interface geometry, binding-condition dependence.
- GB1 single-mutant thermodynamic stability, burial, backbone compatibility.
- General non-viral evidence on stability-mediated evolvability and epistasis, used only with an explicit transfer limitation.

Queries must not request best GB1 mutations, fitness rankings, landscape labels, optima, or benchmark answers.

## Evidence handling

- Give every publication and source span a stable ID.
- Record direct observations separately from cross-source inference.
- Represent conflict as separate claims with scope conditions; never average incompatible findings into false certainty.
- Score scientific credibility and GB1 task applicability separately.
- Permission is always `explanation_only` until independent human review.
- Retrieval similarity is generated at runtime and must not appear as evidence quality.

## Stopping rule

Stop searching when each planned decision card has at least one direct primary or official structural source, its most important boundary is supported, the source ledger is deduplicated by DOI/PDB ID, and two consecutive query families add no new actionable rule. A benchmark failure can reopen only a documented evidence gap; it cannot authorize label lookup.
