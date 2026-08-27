# Source quarantine

Apply this list before query execution, ingestion, card authoring, and indexing.

## Hard exclusions

- OpenAlex and workflows whose primary route is OpenAlex.
- Any virus-protein source or material.
- Any source containing benchmark mutation labels, measured fitness tables, known optima, or test-set recommendations.
- Any FLIP, ProteinGym, Hugging Face, repository, notebook, or mirror that republishes the GB1 four-position binding landscape.
- DOI `10.7554/elife.16965`.
- DOI `10.1016/j.cub.2014.09.072`.
- DOI `10.1073/pnas.1901979116`.
- Any file denied by the repository's active AvoidRead policy.

## Quarantine behavior

When a search provider returns an excluded result, store only:

- an opaque local result identifier;
- provider and timestamp;
- exclusion category;
- whether the result was opened (`false` is required for known leakage sources).

Do not retain the title, abstract, snippet, sequence, mutations, numeric labels, or recommendations. Do not use a quarantined result to formulate a decision card or choose the next query.

## Leakage-safe admissible evidence

Admissible evidence may describe the non-viral GB1 domain's fold, structural regions, independently measured thermodynamic stability, general Fc-interface geometry, assay dependence, residue-class constraints, and transferable directed-evolution principles. It must not disclose benchmark outcomes for candidate variants.
