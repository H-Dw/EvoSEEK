#!/usr/bin/env python3
"""Download datasets declared in configs/data/datasets.yaml.

Examples
--------
    # list registered datasets and profiles
    python scripts/data/download_profile.py --list

    # download the core profile (GB1 + AAV + ProteinGym MVP assays)
    python scripts/data/download_profile.py --profile core

    # download a single dataset, or a glob of them
    python scripts/data/download_profile.py --dataset flip2_priority
    python scripts/data/download_profile.py --dataset "proteingym_*"

    # re-verify local files without network access
    python scripts/data/download_profile.py --profile core --verify-only
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fitness_agents.config import project_root
from fitness_agents.data.download import (
    PROFILE_DIR,
    DownloadError,
    load_registry,
    resolve_profile,
    run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", help="profile name under configs/data/profiles")
    parser.add_argument("--dataset", action="append", default=[],
                        help="dataset id or glob; repeatable")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--offline", action="store_true",
                        help="never touch the network; fail on missing files")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify checksums of local files only")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--list", action="store_true",
                        help="list registered datasets and profiles, then exit")
    return parser.parse_args()


def list_registered(root: Path) -> None:
    registry = load_registry(root)
    print("Datasets:")
    for dataset_id, cfg in sorted(registry.items()):
        print(f"  {dataset_id:34s} {cfg.get('description', '')}")
    print("Profiles:")
    for path in sorted((root / PROFILE_DIR).glob("*.yaml")):
        try:
            ids = resolve_profile(path.stem, root)
        except DownloadError:
            ids = []
        print(f"  {path.stem:12s} -> {', '.join(ids)}")


def main() -> int:
    args = parse_args()
    root = project_root()
    if args.list:
        list_registered(root)
        return 0

    dataset_ids: list[str] = []
    if args.profile:
        dataset_ids.extend(resolve_profile(args.profile, root))
    if args.dataset:
        registry = load_registry(root)
        for pattern in args.dataset:
            if pattern in registry:
                dataset_ids.append(pattern)
            else:
                matched = [d for d in registry if fnmatch.fnmatch(d, pattern)]
                if not matched:
                    print(f"error: no dataset matches {pattern!r}", file=sys.stderr)
                    return 2
                dataset_ids.extend(sorted(matched))
    if not dataset_ids:
        print("error: pass --profile and/or --dataset (or --list)", file=sys.stderr)
        return 2
    dataset_ids = list(dict.fromkeys(dataset_ids))

    results = run(
        dataset_ids, root,
        force=args.force, offline=args.offline,
        verify_only=args.verify_only, retries=args.retries, timeout=args.timeout,
    )
    summary = {
        r.dataset_id: {
            "status": r.status,
            "dest": r.dest,
            "files": [
                {"name": f.name, "status": f.status, "sha256": f.sha256,
                 "sha256_pinned": f.sha256_pinned,
                 "extracted": len(f.extracted_members),
                 **({"message": f.message} if f.message else {})}
                for f in r.files
            ],
        }
        for r in results
    }
    print(json.dumps(summary, indent=2))
    failed = [r.dataset_id for r in results if r.status == "failed"]
    if failed:
        print(f"FAILED datasets: {failed}", file=sys.stderr)
        return 1
    tofu = [f.name for r in results for f in r.files
            if f.status in {"downloaded", "reused"} and not f.sha256_pinned]
    if tofu:
        print(f"note: {len(tofu)} file(s) used trust-on-first-use checksums; "
              "pin them in configs/data/datasets.yaml", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
