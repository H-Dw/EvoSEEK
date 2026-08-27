# Evidence product schema

## Current versions

- Bundle: `scientific-evidence-product:v2`
- Canonical publication: `scientific-publication:v3`
- Release manifest: `scientific-knowledge-release:v2`
- Policy receipt: `external-policy-receipt:v2`
- Review receipt: `evidence-review-receipt:v2`
- Release approval: `evidence-release-approval:v2`

The legacy atomic-claim and publication-catalog v1 files are generated compatibility
views. They are not canonical authoring schemas.

## Release graph

```text
ResearchBrief
  └─ SearchRun
      └─ AllowedSearchHit ── ScopeAssertion
          └─ Publication ─── ScopeAssertion
              └─ SourceSpan ─ ScopeAssertion
                  └─ EvidenceGroup
                      └─ AtomicClaim
                          └─ LogicUnit ─ SearchRun coverage
                              └─ KnowledgeDecisionCard

reviewed record ─ ReviewReceipt
all pre-approval records ─ ReleaseApprovalReceipt
all records and dependencies ─ ReleaseManifest
```

`ScopeAssertion` and `AllowedSearchHit` are independent bundle arrays and explicit
release-record types. Every released card must have an unbroken dependency path to
the brief, exact search execution, accepted metadata observation, source identity,
and source span.

## Record responsibilities

### ResearchBrief

Freezes the question, decision use, QuestionLeaf tree, inclusion/exclusion criteria,
required source types and routes, policy hash, budget, cutoff, source-quarantine IDs,
and stop conditions. It is policy-bound control data.

### SearchRun

Records one provider/query/route execution: exact query and SHA-256, filters, time,
opaque result IDs, accepted result IDs, excluded/duplicate counts, attempt count,
opaque provider-error code, stop reason, snapshot hash, and an exact-query-bound
PolicyReceipt. SearchRun is audit-only and cannot enter runtime retrieval.

### ScopeAssertion

`external-scope-assertion:v1` is keyed by `scope_assertion_id` and exact
`artifact_id`. It records subject scope, roles, explicit absence of the excluded
subject, status, issuer identity/kind, durable verification hash, review/expiry time,
and one or more `review_receipt_ids`. Release requires a verified, allowed assertion
and a passed HMAC-attested human `scope_assertion` ReviewReceipt.

### AllowedSearchHit

`allowed-search-hit:v1` preserves one retained metadata observation with
`search_hit_id`, SearchRun/result identity, disposition, artifact/provider identity,
metadata, retrieval score/components, `scope_assertion_id`, and a signed search-hit
PolicyReceipt. Retrieval score means query relevance only.

An accepted hit must appear in its SearchRun's accepted result set. A
`duplicate_publication` hit remains auditable but cannot be the canonical publication
hit.

### Publication

`scientific-publication:v3` stores normalized identity, study-family identity,
version status, metadata/full-text verification, source scope,
`scope_assertion_id`, `review_receipt_ids`, `canonical_search_hit_id`, and one or more
`acquisitions`, each containing an `AllowedSearchHit.search_hit_id`.

The canonical hit must be accepted; every acquisition must belong to a retained hit
whose normalized identity matches the Publication. Required reviews are
`metadata_identity` and `full_text_scope`; full-text scope requires a human receipt.

### SourceSpan

Pins Publication, artifact and ScopeAssertion IDs, artifact and normalized-span
hashes, resolvable locator, minimal text or permission-safe paraphrase, evidence role,
extraction method, and independent-check flags. `review_receipt_ids` must include a
passed, human-co-signed `source_span_resolution` review over the final dependency
closure.

### EvidenceGroup

Combines the minimum sufficient spans for one proposition and stores stance,
completeness, study independence, grouping rationale, verifier IDs, and
`review_receipt_ids`. A human-co-signed `independence_grouping` review prevents
multiple versions or dependent analyses from inflating independent support.

### AtomicClaim

`scientific-atomic-claim:v2` contains one falsifiable English proposition,
subject–predicate–object fields, knowledge type, applicability, EvidenceGroup links,
and supported/contested/insufficient status. It contains neither retrieval score nor
decision permission.

### LogicUnit

Is the smallest retrieval-sized reasoning unit. It links premise/counterclaim IDs and
the SearchRuns that establish route coverage, then stores operator, bounded
conclusion, applicability tests, falsifiers, abstention conditions, retrieval text,
`scientific_quality`, `task_applicability`, and `review_receipt_ids`. Claim entailment
and task applicability each require a trusted human co-sign.

`retrieval_text` is a discriminative runtime field, not a restatement of the broad
research question. It must name the proposition, applicable context, evidence role,
required observable or feature input, and the boundary that separates it from nearby
LogicUnits. A QuestionLeaf is complete only when its need can be issued as a bounded
retrieval request and its support route is paired with counterevidence or boundary
coverage.

### KnowledgeDecisionCard

Connects LogicUnits to a QuestionLeaf and task route. It stores required inputs,
candidate feature, expected direction, boundary conditions, uncertainty, explicit
permission, calibration/benchmark status, abstention rules, and review/approval IDs.
Any permission above `explanation_only` requires a human `decision_permission`
receipt; selection permissions additionally fail closed under the current schema.

`required_inputs` and `candidate_feature` are runtime routing fields. They must map to
controlled feature channels/focus values or named measurement inputs; free-form tool
names and implicit prerequisites are invalid. `abstain_if` must cover unavailable,
low-quality, non-transferable, or conflicting required inputs.

### ReviewReceipt

`evidence-review-receipt:v2` records review type, reviewed record, reviewer identity
and kind, method version, decision, timezone-aware `reviewed_at` and `expires_at`,
and an `input_sha256` computed over the record's transitive dependency closure,
policy hash, review type, and method version. `expires_at` must be later than
`reviewed_at`. Model-assisted receipts also require model and prompt fingerprints.

Every receipt carries a trusted `canonical-hmac-attestation:v1` with purpose
`evidence-review:v1`. It must be valid at the manifest creation time; without a
manifest, validation uses the latest approval target time or current UTC time.
Editing the reviewed record, any dependency, policy, method, validity window, or
receipt payload makes it stale. Critical review types require a separate human
receipt; the human identity must differ from a model-assisted reviewer identity.

### ReleaseApprovalReceipt

`evidence-release-approval:v2` is human-only. Its signed payload binds the durable
`approval_artifact_id`, exact artifact SHA-256, complete pre-approval
record/dependency set, validator and policy, runtime record allowlist,
excluded-result count, and exact intended release: `release_version`,
`status: released`, timezone-aware `created_at`, and `parent_release_id`. It carries
HMAC purpose `release-approval:v1` and cannot be replayed across an artifact or
release-core change.

### ReleaseManifest

`scientific-knowledge-release:v2` pins every record ID/type/hash/dependency, the
dependency-graph hash, policy and validator versions, status, version/time/parent,
runtime-visible record types, denied-operation count, excluded-result count, and
release-approval IDs. The fixed runtime allowlist is `atomic_claim`, `logic_unit`, and
`knowledge_decision_card`; audit/source records do not enter runtime retrieval.

## Four-way separation

- `AllowedSearchHit.retrieval_score` represents query relevance only.
- `LogicUnit.scientific_quality` represents source, verification, and entailment.
- `LogicUnit.task_applicability` represents context match and decision usefulness.
- `KnowledgeDecisionCard.permission` represents what the runtime may do.

No score or flag may be copied across these layers.
