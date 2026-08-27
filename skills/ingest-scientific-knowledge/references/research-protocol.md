# Upstream research provenance protocol

This file defines what the compatibility exporter must receive from
`$deep-research-evidence`. The ingest skill performs no discovery, browsing,
downloading, source parsing, claim synthesis, or review.

## Required evidence closure

Every exported supported claim must have this canonical closure inside a released
Bundle v2:

`ResearchBrief → SearchRun → AllowedSearchHit → Publication v3 → SourceSpan → EvidenceGroup → AtomicClaim v2 → LogicUnit → KnowledgeDecisionCard`

The same release graph also contains each retained `ScopeAssertion`, all referenced
`ReviewReceipt` records, human `ReleaseApprovalReceipt` records, and ReleaseManifest
v2. SearchRun, ScopeAssertion, AllowedSearchHit, SourceSpan, and receipt records remain
audit-only; the runtime projection does not index them.

## Acceptance checks

Before export, require:

1. The active policy hash equals the ResearchBrief and ReleaseManifest policy hash.
2. Every query and retained metadata observation has an exact-subject PolicyReceipt
   with a trusted HMAC attestation.
3. Every canonical Publication v3 points through `canonical_search_hit_id` and
   `acquisitions` to retained hits; the canonical hit is accepted by its SearchRun.
4. Publication, hit, and SourceSpan share the exact artifact-bound
   `scope_assertion_id`; the assertion set exactly covers retained evidence.
5. Publication identity and full text are verified, each SourceSpan is resolvable and
   independently checked, and dependent analyses share a study family.
6. Scope, span resolution, evidence independence, entailment, applicability, and any
   elevated permission have their required trusted human co-signs. Every receipt has
   a timezone-aware validity window covering release creation time. A model-assisted
   receipt never substitutes for the human receipt.
7. Supported claims have complete supporting EvidenceGroups and reviewed LogicUnit
   entailment; contested claims are qualified and insufficient claims lead only to
   explicit abstention.
8. Counterevidence routes are complete and linked findings agree with the declared
   counterevidence status.
9. Benchmark/test-source identifiers are quarantined before source use. Prohibited,
   mixed, unknown, expired, or conflicting source scope is absent from the release.
10. The human ReleaseApprovalReceipt verifies against the approver keyring and binds
    the durable approval-artifact ID and hash, full pre-approval record closure, and
    exact release version, `released` status, timezone-aware creation time, and
    parent release ID.

Any failed condition blocks export. Ingest does not repair provenance by editing the
v1 projection; return to the canonical evidence product, regenerate affected reviews
and approval, and create a new release.

## Projection rules

- Export only canonical AtomicClaims with `claim_status: supported`.
- Derive confidence from reviewed scientific quality and uncertainty, never from
  retrieval score.
- Derive citations only from supporting EvidenceGroups and released SourceSpans.
- Preserve `source_release_id` and canonical `source_record_hash` on every claim.
- Preserve release identity in publication verification metadata.
- Force `selection_eligible: false`; canonical permissions are not representable in
  the v1 claim format.
- Publish atomically to a new directory, then validate only the exact
  `runtime-files.json` file list.
