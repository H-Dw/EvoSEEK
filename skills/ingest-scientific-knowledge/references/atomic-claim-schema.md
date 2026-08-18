# Atomic claim and publication schema

Each claim is an English Markdown file with YAML front matter using
`scientific-atomic-claim:v1`. Required fields are `record_type: atomic_claim`, stable
`claim_id`, `language: en`, snake-case `knowledge_type`, one-sentence `statement`,
`subject`, `predicate`, `object`, `polarity`, `claim_kind`, confidence in `[0, 1]`,
structured `applicability`, one or more `citation_support` mappings, and
`selection_eligible` (normally false).

The Markdown body contains the same atomic statement, with at most concise scope or
limitation text. It contains no bibliography, prompt instruction, or independent
second recommendation.

The catalog uses `scientific-publications:v1`. A DOI-backed record uses a lowercase
`publication_id` of `doi:<doi>` and stores title, authors, year, venue, DOI, canonical
URL, publication type, and independent verification metadata.

The normalized KG path is:

`Document -> HAS_CHUNK -> DocumentChunk -> ASSERTS -> Claim`

`Claim -> SUPPORTED_BY_CITATION -> CitationSupport -> CITES_PUBLICATION -> Publication`

`CitationSupport -> DERIVED_FROM -> DocumentChunk`

Raw chunks remain contextual and non-selecting. Candidate selection requires a
separate validated projection that creates candidate-specific `Evidence`.
