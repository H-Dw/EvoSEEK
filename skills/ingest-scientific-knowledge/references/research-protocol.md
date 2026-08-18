# Scientific research and verification protocol

## Discovery

Translate the objective into English concept blocks: mechanism, system, intervention,
outcome, and limitation. Combine synonyms within a block and vary one block at a
time. Search at least one bibliographic index and one primary publisher or repository
route. Record the exact query, search date, and filters.

Recommended route order:

1. Crossref, PubMed, OpenAlex, or Semantic Scholar for discovery and DOI resolution.
2. Publisher landing page, PubMed Central, Europe PMC, arXiv, ACL Anthology, or an
   institutional repository for source verification.
3. Citation chaining for influential prior work and direct counterevidence.

## Two independent checks

Publication verification checks title, author list, year, venue, DOI, and URL.
Claim-support verification checks whether the cited source actually entails the
atomic statement in its scope. Record these independently.

Use these `support_type` values:

- `direct_support`: data or analysis directly supports the statement.
- `empirical_example`: one study demonstrates an instance, not universality.
- `method_basis`: the source defines or validates the named method.
- `background_support`: a review supports a broad statement.
- `limiting`: the source narrows or contradicts the statement.

Use a resolvable locator such as `abstract`, `figure_2`, `table_1`, `results`, or
`title_and_method_scope`. Never mark a locator verified if only metadata was read.

## Synthesis gates

Before saving a claim, confirm that it is one falsifiable proposition, excludes
target-specific labels, states applicability, separates scientific confidence from
retrieval rank, references cataloged publications, and includes a limitation or
counterevidence search. Otherwise keep it in research notes rather than the corpus.
