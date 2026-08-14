#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST_DIR="${ROOT_DIR}/data/raw/flip/gb1"
COMMIT="62cace8735f5610e2743cf06ce0f944b37fffaa6"
ARCHIVE_URL="https://raw.githubusercontent.com/facebookresearch/FLIP/${COMMIT}/splits/gb1/four_mutations_full_data.csv.zip"
README_URL="https://raw.githubusercontent.com/facebookresearch/FLIP/${COMMIT}/splits/gb1/README.md"
EXPECTED_SHA256="85692d808dcd3ae54fa2ac31f4e590858d4582369b6c7b05df299b9b6c383bff"

mkdir -p "${DEST_DIR}"
curl --fail --location --retry 3 "${ARCHIVE_URL}" --output "${DEST_DIR}/four_mutations_full_data.csv.zip"
curl --fail --location --retry 3 "${README_URL}" --output "${DEST_DIR}/UPSTREAM_README.md"

ACTUAL_SHA256="$(sha256sum "${DEST_DIR}/four_mutations_full_data.csv.zip" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  echo "Checksum mismatch: expected ${EXPECTED_SHA256}, got ${ACTUAL_SHA256}" >&2
  exit 1
fi

python3 - "${DEST_DIR}/four_mutations_full_data.csv.zip" "${DEST_DIR}" <<'PY'
from pathlib import Path
import sys
import zipfile

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
with zipfile.ZipFile(archive) as handle:
    members = [name for name in handle.namelist() if name.endswith(".csv")]
    if members != ["four_mutations_full_data.csv"]:
        raise SystemExit(f"Unexpected archive members: {members}")
    handle.extract(members[0], destination)
print(destination / members[0])
PY

echo "FLIP GB1 downloaded and verified at ${DEST_DIR}"

