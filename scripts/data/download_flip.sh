#!/usr/bin/env bash
# Download the FLIP sources used by this project: GB1 + AAV (+ active splits).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

EXTRA_ARGS=()
if [[ "${FITNESS_AGENTS_FORCE_DOWNLOAD:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--force")
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/data/download_profile.py" \
  --dataset flip_gb1 --dataset flip_aav --dataset flip_aav_splits \
  "${EXTRA_ARGS[@]}" "$@"
