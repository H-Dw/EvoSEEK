#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_PROFILE="${1:-dev}"

"${PYTHON_BIN}" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(f"Python 3.10-3.13 required, found {sys.version.split()[0]}")
PY

"${PYTHON_BIN}" -m venv "${ROOT_DIR}/.venv"
"${ROOT_DIR}/.venv/bin/python" -m pip install --upgrade pip
case "${INSTALL_PROFILE}" in
  base) "${ROOT_DIR}/.venv/bin/python" -m pip install -r "${ROOT_DIR}/requirements/base.txt" ;;
  dev) "${ROOT_DIR}/.venv/bin/python" -m pip install -r "${ROOT_DIR}/requirements/dev.txt" ;;
  llm) "${ROOT_DIR}/.venv/bin/python" -m pip install -e "${ROOT_DIR}[dev,llm]" ;;
  *) echo "Usage: bash scripts/setup_linux.sh [base|dev|llm]" >&2; exit 2 ;;
esac
"${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/check_environment.py"

