# GB1 feature-provider example resources

- `1PGB.cif` is the official RCSB mmCIF download for the unmutated 56-residue
  protein G B1 domain (chain A, 1.92 Å). Source:
  <https://files.rcsb.org/download/1PGB.cif>; entry metadata:
  <https://www.rcsb.org/structure/1PGB>.
- `rcsb_pdb_1PGB.fasta` is the configured reference-sequence input.
- `msa/0/non_pairing.a3m` is the configured single-chain conservation input.
  With the example provider filters it retains 39 sequences with Neff about
  15.13 and complete coverage at positions 39, 40, 41, and 54.
- `msa/0/pairing.a3m` is retained for paired-MSA workflows; it is not the
  primary single-chain conservation input. `msa/0/hmmsearch.a3m` contains
  structural-template hits and is not used as the conservation MSA.
- `gb1_precomputed.a3m` remains a minimal two-sequence format example only.

The FASTA and structure resource are configured in
`configs/task/gb1_binding_features.example.yaml`; the selected MSA and provider
thresholds are configured in `configs/knowledge/gb1_features.example.yaml`.
Replace the example A3M with a deeper target-appropriate alignment before
treating its conservation profile as strong evolutionary evidence.
