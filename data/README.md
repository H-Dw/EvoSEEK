# Data layout and licenses

Run `bash scripts/data/download_flip_gb1.sh` followed by
`python scripts/data/prepare_gb1.py`.

- `raw/flip/gb1/`: upstream FLIP archive and extracted CSV.
- `processed/gb1_full_public.csv`: candidate metadata without labels.
- `processed/gb1_full_oracle.csv`: simulation-only labels.
- `demo/gb1_demo_public.csv`: 512-row CPU demo metadata without hidden labels.
- `demo/gb1_demo_oracle.csv`: simulation-only demo labels.

The original GB1 measurements are CC BY 4.0. FLIP-derived data and splits are AFL-3.0. Keep
the generated manifest and upstream README with any redistribution. The separate public/oracle
files are an engineering boundary for leakage testing, not a cryptographic security mechanism.

