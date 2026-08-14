#!/usr/bin/env python3
"""Validate downloaded raw datasets against the registry.

Checks, per dataset in scope:
- every required file exists and matches its pinned sha256 (when pinned);
- a download_manifest.json exists and agrees with the files on disk;
- extracted CSVs meet the registry's expected_min_rows lower bounds.

Exit code 0 = all good, 1 = at least one failure, 2 = usage error.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fitness_agents.config import project_root
from fitness_agents.data.download import (
    MANIFEST_NAME,
    load_registry,
    resolve_profile,
    verify_dataset,
)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        return sum(1 for _ in reader)


def validate(dataset_ids: list[str], root: Path) -> dict:
    registry = load_registry(root)
    report: dict[str, dict] = {}
    for dataset_id in dataset_ids:
        cfg = registry[dataset_id]
        dest = Path(cfg["dest"])
        if not dest.is_absolute():
            dest = root / dest
        entry: dict[str, object] = {"dest": str(dest), "checks": [], "valid": True}

        def check(ok: bool, label: str) -> None:
            entry["checks"].append({"ok": ok, "check": label})
            if not ok:
                entry["valid"] = False

        # 1. file presence + pinned checksums (engine verify pass)
        verify = verify_dataset(dataset_id, cfg, root)
        for f in verify.files:
            check(f.status in {"verified", "skipped_optional"},
                  f"file {f.name}: {f.status}{' - ' + f.message if f.message else ''}")

        # 2. manifest presence (skipped for never-downloaded datasets)
        manifest_path = dest / MANIFEST_NAME
        any_local = any((dest / fc.get("target", fc["name"])).exists()
                        for fc in cfg.get("files") or [])
        check(manifest_path.exists() or not any_local,
              f"manifest {MANIFEST_NAME} present")

        # 3. expected_min_rows on extracted CSVs
        for csv_name, min_rows in (cfg.get("expected_min_rows") or {}).items():
            matches = list(dest.rglob(csv_name))
            if not matches:
                check(False, f"{csv_name}: not found under {dest}")
                continue
            rows = count_csv_rows(matches[0])
            check(rows >= int(min_rows), f"{csv_name}: {rows} rows >= {min_rows}")

        report[dataset_id] = entry
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="validate a download profile")
    parser.add_argument("--dataset", action="append", default=[],
                        help="dataset id or glob; repeatable")
    args = parser.parse_args()
    root = project_root()

    dataset_ids: list[str] = []
    if args.profile:
        dataset_ids.extend(resolve_profile(args.profile, root))
    registry = load_registry(root)
    for pattern in args.dataset:
        if pattern in registry:
            dataset_ids.append(pattern)
        else:
            dataset_ids.extend(sorted(d for d in registry if fnmatch.fnmatch(d, pattern)))
    if not dataset_ids:
        dataset_ids = sorted(registry)  # default: validate everything
    dataset_ids = list(dict.fromkeys(dataset_ids))

    report = validate(dataset_ids, root)
    print(json.dumps(report, indent=2))
    invalid = [d for d, r in report.items() if not r["valid"]]
    if invalid:
        print(f"INVALID: {invalid}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
