# Generated native evidence records and atomic-claim compatibility schema

## Canonical source versus runtime projection

The source of truth is a released `scientific-evidence-product:v2` with canonical
`scientific-publication:v3` records and a
`scientific-knowledge-release:v2` manifest. The exporter generates:

- English Markdown claims using `scientific-atomic-claim:v1`;
- `catalog/publications.yaml` using `scientific-publications:v1`;
- canonical `evidence-release.json` containing the complete Bundle v2, including
  signed review/approval receipts and its Release v2 manifest; and
- `runtime-files.json` using `local-rag-runtime-files:v1`.

The default runtime projection uses `local-rag-runtime-files:v2` and contains
canonical-hash-bound AtomicClaim, LogicUnit, and KnowledgeDecisionCard Markdown
records. It preserves discriminative retrieval text, scientific quality, task
applicability, boundaries, counterclaims, required inputs, feature routing,
permission, abstention, and controlled facets. The v1 claim/catalog files are a
lossy compatibility view. Do not hand-author them,
copy a template into a corpus, or promote a search summary into either format.

## Generated claim file

Each Markdown file has YAML front matter with these required fields:

- `schema_version: scientific-atomic-claim:v1`
- `record_type: atomic_claim`
- stable `claim_id`, `title`, and `language: en`
- snake-case `knowledge_type`
- one-sentence `statement`, `subject`, `predicate`, `object`, `polarity`, and
  `claim_kind`
- numeric scientific `confidence` in `[0, 1]`, derived from reviewed LogicUnits—not
  search rank or retrieval similarity
- structured `applicability`
- one or more `citation_support` mappings
- `selection_eligible: false`
- `source_release_id` equal to the exact Release v2 ID
- `source_record_hash` equal to the canonical AtomicClaim hash in Release v2

The Markdown body must equal the normalized atomic statement exactly. It contains no
bibliography, extra recommendation, instruction, secret, executable directive,
target measurement, or language mixing.

Each citation support has stable `support_id`, DOI-backed `publication_id`, one of
`direct_support`, `empirical_example`, `method_basis`, `background_support`, or
`limiting`, a resolvable `locator`, and `verified_against_source: true`. The exporter
only emits citation support derived from released, resolved SourceSpans.

## Generated publication catalog

`scientific-publications:v1` contains `generated_from` (the exact Release v2 ID),
`verified_on`, and DOI-backed publications. Each publication stores title, authors,
year, venue, DOI, canonical URL, publication type, and a verification block with:

- `metadata_source: deep-research-evidence-product`
- `metadata_verified: true`
- `full_text_verified: true`
- `source_release_id` matching `generated_from`

Canonical Publication v3 retains `canonical_search_hit_id`, acquisitions,
`scope_assertion_id`, and `review_receipt_ids` in `evidence-release.json`. Those
fields are deliberately not flattened into the v1 catalog; the exporter verifies
their complete release provenance before projection.

## Runtime manifest

`runtime-files.json` is the only file-discovery authority. It pins source release and
policy hashes, workspace access-policy hash, each relative path/record identity/hash/
byte count, and its own `manifest_sha256`. Released-bundle validation reads only
those exact files and rejects recursive discovery, extra inferred files, path
escapes, symlinks/junctions, and content mismatches.

The exact `evidence-release.json` bytes are also manifest-addressed. Runtime loading
requires the explicit policy/reviewer/approver trust configuration and revalidates
the complete EvidenceProduct; ordinary file hashes or non-empty approval IDs cannot
self-authenticate a rewritten bundle. It then deterministically rerenders the
supported-claim and publication-catalog projection from that signed bundle and
requires the on-disk bytes and file set to match exactly. Older detached-manifest v1
exports must be regenerated.

The normalized KG compatibility path is:

`Document → HAS_CHUNK → DocumentChunk → ASSERTS → Claim`

`Claim → SUPPORTED_BY_CITATION → CitationSupport → CITES_PUBLICATION → Publication`

`CitationSupport → DERIVED_FROM → DocumentChunk`

Raw chunks remain contextual and non-selecting. Candidate selection requires a
separate canonical, calibrated, manifest-addressed projection; the current exporter
always writes `selection_eligible: false`.
