## Shared ID Integrity Protocol

When the request supplies an `id_maps`, `sample_map`, `evidence_map`, or other allowed-ID region:

1. Treat every short label as an opaque identifier local to this logical request. Copy it
   character-for-character; never translate, expand, shorten, concatenate, renumber, or infer
   priority or scientific meaning from it.
2. Use only labels present in the current request's allowed-ID region. Never invent a label and
   never reconstruct or return a hidden canonical long identifier.
3. Put identifiers in the schema fields intended for identifiers. Mentioning one only in prose
   does not satisfy required coverage or citation closure.
4. Return every schema-required sample exactly once. Do not omit, duplicate, or substitute an ID,
   and do not repair a damaged ID by guessing a similar-looking label.
5. When no allowed evidence or fact label supports a statement, use the schema's typed limitation,
   unresolved, or missing-evidence form. Do not borrow an ID from another sample or namespace.

These constraints apply independently to every LLM transaction. Labels from an earlier request
have no identity in the current request unless they are explicitly supplied again.
