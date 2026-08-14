# Data layout and licenses

Downloads are registry-driven: `configs/data/datasets.yaml` declares every
source (pinned version, mirror URLs, sha256, license, adapter), and
`configs/data/profiles/*.yaml` bundles datasets into download profiles.

## Quick start

```bash
# list registered datasets and profiles
python scripts/data/download_profile.py --list

# smoke test (~3 MB): FLIP GB1 only
python scripts/data/download_profile.py --profile smoke

# core validation set (~1.2 GB): GB1 + AAV (+ active splits) + ProteinGym MVP assays
python scripts/data/download_profile.py --profile core

# evidence (+6 GB): AF2 structures + MVP-assay MSAs for structure/conservation ablations
python scripts/data/download_profile.py --profile evidence

# extended: core + all 16 FLIP2 splits (OOD generalization)
python scripts/data/download_profile.py --profile extended

# full local mirror (~8 GB)
python scripts/data/download_profile.py --profile full

# verify local files without network access
python scripts/data/validate_downloads.py --profile core
```

Per-source wrappers: `download_flip.sh`, `download_flip2.py [--priority]`,
`download_proteingym.py [--scope mvp|full|cv-folds|indels]`,
`download_msas.py [--scope mvp|full]`, `download_structures.py`.
Common flags: `--force`, `--offline`, `--verify-only`.

## Profiles

| profile | contents | approx. size |
|---|---|---|
| smoke | FLIP GB1 | 3 MB |
| core | smoke + FLIP AAV (+ active splits) + ProteinGym metadata + 7 MVP assays | 1.2 GB |
| evidence | core + AF2 structures + MVP-assay MSAs | 6.5 GB |
| extended | core + FLIP2 (7 datasets, 16 splits) | 1.5 GB |
| full | everything incl. full substitutions, CV folds, indels, all MSAs | ~8 GB |

MVP assays (`configs/data/proteingym_mvp_assays.txt`): SPG1 (GB1), GFP, AAV2
capsid, TEM-1 Stiffler + Firnberg, Spike binding + expression. FLIP AAV uses
the active (green) splits only: one/two/seven_vs_many, low_vs_high, des_mut,
mut_des.

## Guarantees

- Pinned upstream versions (FLIP commit `62cace8`, ProteinGym v1.3 / commit
  `144fe22`, FLIP2 2025 release); primary URLs never track a moving branch.
- Mirror fallback, resume via `.partial` + HTTP Range, atomic rename after
  checksum, zip-member whitelists with path-traversal guards.
- sha256 pinned in the registry where known; unknown files are downloaded
  trust-on-first-use and the digest is recorded in
  `<dest>/download_manifest.json` — pin it back into `datasets.yaml`.
- Every dataset writes `download_manifest.json` (version, URLs, checksums,
  license, citation, extracted members). `validate_downloads.py` re-checks
  files, manifests and per-assay row-count lower bounds.

## Layout

- `raw/flip/gb1/`, `raw/flip/aav/`: FLIP archives, extracted CSVs, upstream READMEs.
- `raw/flip2/`: FLIP2 split CSVs (gunzipped).
- `raw/proteingym/reference/`: DMS assay metadata tables.
- `raw/proteingym/substitutions/`: substitutions archive + extracted assay CSVs.
- `raw/proteingym/indels/`, `raw/proteingym/cv_folds/`, `raw/proteingym/msa/`,
  `raw/proteingym/structures/`: extended ProteinGym resources.
- `processed/gb1_full_{public,oracle}.csv`, `demo/gb1_demo_{public,oracle}.csv`:
  built by `python scripts/data/prepare_gb1.py`.

## Licenses

Original GB1 measurements CC BY 4.0; FLIP/FLIP2 splits AFL-3.0 / CC BY 4.0
(Amylase MIT); ProteinGym benchmark CC BY-NC-SA 4.0 with per-assay terms (see
`raw/proteingym/reference/DMS_substitutions.csv`). Keep manifests and upstream
READMEs with any redistribution. The public/oracle file split is an
engineering boundary for leakage testing, not a security mechanism.
