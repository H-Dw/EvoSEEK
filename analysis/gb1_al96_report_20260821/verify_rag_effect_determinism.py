"""Rerun the RAG-effect audit and verify deterministic derived artifacts."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from config import RAG_DIAGNOSTIC_CASE_DIR, RAG_DIAGNOSTIC_DIR, RAG_DIAGNOSTIC_SOURCE_DIR


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict[str, str]:
    files = [RAG_DIAGNOSTIC_DIR / "rag_effect_analysis.md"]
    files.extend(
        path for path in RAG_DIAGNOSTIC_SOURCE_DIR.rglob("*") if path.is_file()
    )
    files.extend(path for path in RAG_DIAGNOSTIC_CASE_DIR.rglob("*") if path.is_file())
    return {path.as_posix(): _sha256(path) for path in sorted(files)}


def main() -> int:
    before = snapshot()
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("run_rag_effect_diagnostics.py"))],
        check=True,
    )
    after = snapshot()
    changed = sorted(
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    )
    print(f"deterministic_rag_effect_files={len(after)} changed={len(changed)}")
    if changed:
        print("\n".join(changed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
