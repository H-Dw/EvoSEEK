"""Rerun the anomaly analysis and verify deterministic analytical outputs."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from config import ANOMALY_DIR, ANOMALY_SOURCE_DATA_DIR


CHECKED_FILES = (
    ANOMALY_DIR / "anomaly_analysis.md",
    ANOMALY_DIR / "figure4_qa.md",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict[str, str]:
    files = [*CHECKED_FILES]
    files.extend(
        path for path in ANOMALY_SOURCE_DATA_DIR.rglob("*") if path.is_file()
    )
    return {path.as_posix(): _sha256(path) for path in sorted(files)}


def main() -> int:
    before = snapshot()
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("run_anomaly_diagnostics.py"))],
        check=True,
    )
    after = snapshot()
    changed = sorted(
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    )
    print(f"deterministic_anomaly_files={len(after)} changed={len(changed)}")
    if changed:
        print("\n".join(changed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
