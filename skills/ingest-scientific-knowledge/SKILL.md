---
name: ingest-scientific-knowledge
description: Export a released deep-research evidence product into native AtomicClaim, LogicUnit, and DecisionCard runtime records, with an optional legacy compatibility view, and validate its exact manifest before local RAG or KG indexing. Use after evidence research and release, not for literature search.
---

# Ingest Scientific Knowledge

Build the native evidence-record view consumed by the Agentic RAG/KG runtime.
Canonical research and review happen upstream in `$deep-research-evidence`; this
skill performs no search and never turns summaries into indexable claims.

The accepted input is a released `scientific-evidence-product:v2` whose canonical
Publication records are `scientific-publication:v3` and whose manifest is
`scientific-knowledge-release:v2`. Legacy v1 claim/catalog formats are lossy,
generated-only compatibility projections.

## Trust prerequisites

Run all commands from the repository root. Export revalidates every policy, review,
approval, and manifest record, so the same four trust variables used to create the
release must be present:

- `FITNESS_DEEP_RESEARCH_POLICY_KEY_ID`
- `FITNESS_DEEP_RESEARCH_POLICY_KEY_HEX`
- `FITNESS_DEEP_RESEARCH_REVIEW_KEYRING_JSON`
- `FITNESS_DEEP_RESEARCH_RELEASE_KEYRING_JSON`

The policy key ID is lowercase and ends in `:vN`; the policy key is at least 32 bytes
as hex. Each keyring is a JSON object keyed by exact reviewer/approver ID:

```json
{"reviewer-or-approver-id":{"key_id":"versioned-signer:v1","key_hex":"<at-least-64-hex-characters>"}}
```

Review and release keyrings may contain multiple signer entries. Keep all real key
material in a secret manager, never in an export or repository. Missing identities,
key-ID mismatch, or invalid HMAC attestations fail closed.

## Required workflow

1. Use `$deep-research-evidence` to produce the complete evidence graph. Require
   independent ScopeAssertion and AllowedSearchHit records, exact-query policy
   receipts, an accepted canonical Publication hit plus retained accepted/duplicate
   acquisition provenance, verified SourceSpans, critical
   human co-signs valid at release creation time, and a human release approval bound to the durable approval
   artifact ID/hash plus exact release version/status/created-at/parent. Prohibited,
   mixed, unknown, and quarantined source scope must not enter the release.
2. Read [references/research-protocol.md](references/research-protocol.md) and
   [references/atomic-claim-schema.md](references/atomic-claim-schema.md). Do not use
   the assets as hand-authoring forms; they document generated shapes only.
3. Revalidate, then export the native runtime to a new, nonexistent directory:

   ```powershell
   python skills/deep-research-evidence/scripts/run_deep_research.py validate <released-bundle.json> --output <release-validation.json>
   python skills/deep-research-evidence/scripts/run_deep_research.py export-native <released-bundle.json> --output-root <new-runtime-bundle-directory>
   ```

   Export is atomic and refuses to overwrite an existing target. It preserves native
   AtomicClaim, LogicUnit, and KnowledgeDecisionCard records, including required
   inputs, feature routing, counterclaims, boundaries, applicability, permission, and
   abstention. It always writes `selection_eligible: false`. Use `export-legacy` only
   when an older consumer explicitly requires atomic claims.
4. Require the generated `runtime-files.json`. Native schema v2 content-addresses all
   exported AtomicClaim/LogicUnit/DecisionCard records and `evidence-release.json`.
   Legacy schema v1 additionally contains a publication catalog. The latter
   is the complete canonical Bundle v2, including ReviewReceipts,
   ReleaseApprovalReceipts, attestations, and ReleaseManifest v2—not a detached
   manifest that can self-assert approval. Only files named there are eligible for
   validation or indexing; recursive discovery is disabled.
5. Validate the exact runtime bundle:

   ```powershell
   python skills/ingest-scientific-knowledge/scripts/validate_knowledge_bundle.py <new-runtime-bundle-directory>
   python skills/ingest-scientific-knowledge/scripts/validate_knowledge_bundle.py <new-runtime-bundle-directory> --embedding-model <local-model-path>
   ```

   The second form is optional and proves every claim fits the actual local
   tokenizer. The model path must be local; the validator does not download models.
6. Index only after zero validation errors. Build the index from the exact current
   `runtime-files.json`; a prior index must be rebuilt from the new manifest, not
   incrementally trusted by directory contents.

## Manifest-only rules

- Missing, malformed, unsigned, or hash-mismatched `runtime-files.json` is a hard
  failure.
- Reject paths not listed in the manifest, duplicate/case-colliding paths, path
  escapes, missing files, symlinks/junctions, byte-count mismatch, and SHA-256
  mismatch.
- Require manifest `source_release_id`, policy hash, and workspace access-policy hash
  to match the embedded Release v2 evidence record.
- Revalidate the complete embedded Bundle v2 with the four explicit trust variables
  on every validation/index load. Missing trust, a modified receipt/signature, or a
  detached ReleaseManifest is a hard failure.
- Require each generated record's `source_release_id` and `source_record_hash` to
  match the exact canonical record entry and type in Release v2.
- Recompute every generated record byte deterministically from the signed
  Bundle v2. Rewriting a projection and recomputing ordinary file/manifest hashes is
  rejected even when its claimed `source_record_hash` is left unchanged.
- For legacy export, require each generated publication's verification block to name the same source
  release and confirm canonical metadata and full-text verification.
- Keep SearchRun, ScopeAssertion, AllowedSearchHit, raw SourceSpan text, and review
  records audit-only; they are not runtime retrieval files.

## Output contract

Return the source release ID, runtime-manifest hash, exact manifest paths, counts by
record/facet/knowledge type, validation output, and tokenizer-check status. Never call an
unmanifested, recursively discovered, hand-authored, or partially verified corpus
production-ready.
