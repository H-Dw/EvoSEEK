# Optional structure service boundary

The MVP uses versioned, precomputed site-risk evidence and does not claim that ipTM is experimental
fitness. A production structure sidecar should implement a narrow endpoint such as:

```text
POST /v1/score
{protein_id, variant_id, sequence, structure_uri, requested_features}

-> {model_version, plddt, ptm, iptm?, pae?, sasa?, interface_contacts?, provenance}
```

`ipTM` is permitted only as an auxiliary complex-plausibility feature. It must not be labeled as
binding affinity or measured fitness. The core package consumes these values through an evidence
provider, so AlphaFold/Boltz/Rosetta implementations can be replaced without changing the loop.

