"""Rerun the analysis and verify deterministic data/table/case outputs."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from config import CASE_STUDY_DIR, SOURCE_DATA_DIR, TABLE_DIR


CHECKED_DIRECTORIES = (SOURCE_DATA_DIR, TABLE_DIR, CASE_STUDY_DIR)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict[str, str]:
    files = sorted(
        path
        for directory in CHECKED_DIRECTORIES
        for path in directory.rglob("*")
        if path.is_file()
    )
    return {path.as_posix(): _sha256(path) for path in files}


def main() -> int:
    before = snapshot()
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("run_analysis.py"))],
        check=True,
    )
    after = snapshot()
    changed = sorted(
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    )
    print(f"deterministic_files={len(after)} changed={len(changed)}")
    if changed:
        print("\n".join(changed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
