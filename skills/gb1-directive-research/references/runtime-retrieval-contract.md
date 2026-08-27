# Runtime Retrieval Contract

Use this contract when a blinded benchmark shows high card recall without fitness improvement.

## Principle

Retrieval success is not evidence of decision value. Minimize injected context and route it to the current evidence gap. Scientific credibility, task applicability, retrieval similarity, and decision permission remain separate fields.

## Agentic evidence routing

- Start from round-visible measurements and the prior hypothesis assessment/reflection.
- State one evidence gap as a complete English scientific question.
- If support is requested, pair it with counterevidence or a boundary request.
- Filter first by controlled facets: record type, scientific proposition/question leaf,
  decision slot, task route, feature channel/focus, required input, permission, expected
  direction, optional stage, and evidence role.
- Retrieve at most the configured record budget, deduplicate by canonical record ID,
  and preserve record permission and abstention conditions.

Queries must name the missing evidence, observable context, intended use, and boundary.
Do not use a generic omnibus query. Stage is only an optional facet and cannot replace
the scientific question.

Semantic similarity does not establish scientific quality, applicability, or permission.
Record the selected record IDs and facets for every query. Fail the routing audit when
support lacks counter/boundary coverage, required inputs are unavailable, permissions
are upgraded, or multiple records are indistinguishable under their retrieval text.

## Iteration rule

The next bundle may use only blinded aggregate feedback. If all intended cards were retrieved but mean AUC declined, reduce context breadth before adding publications. Do not inspect mutation identities, per-variant labels, or successful sequences.
