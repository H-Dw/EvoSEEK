# Quality and release gates

## Deterministic hard gates

Validation fails on:

- wrong versions/types, unknown fields, duplicate IDs, dangling dependencies, hash
  mismatch, dependency cycles, or incomplete release-record coverage;
- any QuestionLeaf missing a declared search route, including counterevidence;
- a SearchRun PolicyReceipt whose canonical HMAC attestation is untrusted, invalid,
  stale, policy-mismatched, or not bound to the exact query;
- a retained SearchHit absent from its SearchRun results, or an accepted hit absent
  from `accepted_result_ids`;
- any `ScopeAssertion` that is unverified, expired, artifact-mismatched, not explicitly
  clear of the excluded subject, outside allowed scope, unreferenced, or lacks its
  trusted human review receipt;
- a ScopeAssertion whose reviewed/expiry window does not cover the exact release
  reference time;
- a SearchHit PolicyReceipt that does not bind its complete retained metadata payload;
- a SearchHit whose result, artifact, provider-record, DOI, URL, or canonical hit
  identity intersects the brief's benchmark/source quarantine;
- a Publication v3 whose canonical hit is not accepted, whose acquisition hit is
  absent/mismatched, or whose `scope_assertion_id` differs from its hit/artifact;
- any result or Publication identity intersecting the benchmark/source quarantine;
- unverified publication identity/full text, unresolved or independently unchecked
  SourceSpan, or a span whose Publication/artifact/ScopeAssertion provenance diverges;
- incomplete EvidenceGroup used by a supported claim, inflated study-family support,
  or missing independence review;
- an insufficient claim used outside an explicit abstention, an unqualified contested
  premise, or a supported claim not covered by reviewed LogicUnit entailment;
- instruction-like content in span paraphrases, claims, or retrieval text;
- unverified claim entailment, missing counterevidence route coverage, or
  `searched_found` without linked limiting/refuting evidence;
- a missing, failed, escalated, stale, identity-mismatched, unauthorized, unsigned,
  untrusted, dependency-incomplete, not-yet-effective, or expired ReviewReceipt;
- a critical review type without a distinct trusted human co-sign;
- candidate-reranking or hard-gate permission under the current schema; calibration
  and benchmark-overlap records are not yet manifest-addressable;
- a released manifest without a trusted human ReleaseApprovalReceipt bound to the
  complete product and exact version/status/created-at/parent intent;
- a ReleaseApprovalReceipt with a missing durable artifact ID, placeholder artifact
  hash, duplicate approver, altered input closure, wrong key ID, invalid HMAC, or
  target-core mismatch;
- any denied-path operation.
- a runtime compatibility export containing only a detached ReleaseManifest rather
  than the complete signed Bundle v2, or loaded without its explicit trust keyrings.

## Review authorization matrix

| Review type | Allowed proposer/checker | Human co-sign required |
| --- | --- | --- |
| `metadata_identity` | human or deterministic rule | no |
| `full_text_scope` | human | yes |
| `scope_assertion` | human | yes |
| `source_span_resolution` | human or model-assisted | yes |
| `independence_grouping` | human or model-assisted | yes |
| `claim_entailment` | human or model-assisted | yes |
| `task_applicability` | human or model-assisted | yes |
| `decision_permission` | human | yes |

A model-assisted receipt requires both `model_fingerprint` and `prompt_sha256`. When
one exists for a critical review, a different human reviewer identity must co-sign
the same review type and record. Every accepted receipt must be `passed`, point back
from the record's `review_receipt_ids`, match its dependency-closure
`input_sha256`, and verify against the exact reviewer key in
`FITNESS_DEEP_RESEARCH_REVIEW_KEYRING_JSON`. Both `reviewed_at` and `expires_at` are
timezone-aware, `expires_at` follows `reviewed_at`, and the release/reference time is
inside that inclusive window. The reference is manifest `created_at`, otherwise the
latest approval target time, otherwise current UTC time.

## Attestation gates

- Policy receipts are `external-policy-receipt:v2`, signed with purpose
  `policy-receipt:v1`, and verified by the exact active policy key.
- Review receipts are `evidence-review-receipt:v2`, signed with purpose
  `evidence-review:v1`, and verified by reviewer identity plus versioned key ID.
- Release approvals are `evidence-release-approval:v2`, signed with purpose
  `release-approval:v1`, and verified by human approver identity plus versioned key
  ID.
- Unknown key IDs, empty keyrings, keys shorter than 32 bytes, unsupported key/purpose
  versions, noncanonical payloads, altered payload hashes, and invalid signatures all
  fail closed.

HMAC establishes authenticity only inside the explicitly provisioned shared-secret
trust boundary. Protect key material and rotate by creating a new `:vN` key ID; never
silently reuse another key version.

## Permission ladder

1. `explanation_only`: default; verified evidence may explain.
2. `hypothesis_generation`: requires reviewed evidence and human permission review;
   may propose a testable hypothesis, not rerank.
3. `candidate_reranking`: rejected by the current validator until a canonical,
   manifest-addressed calibration/overlap record type exists.
4. `hard_gate`: rejected for the same reason and additionally requires two distinct
   human approvals and a fail-safe fallback.

The generated v1 compatibility view always sets `selection_eligible: false`; it never
converts a canonical card permission into a claim-level selection flag.

Downgrade permission whenever a dependency is corrected, retracted, stale, disputed,
or outside applicability. Never mutate a released manifest; generate new receipts,
approval, and a child release.
