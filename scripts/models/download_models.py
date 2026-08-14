#!/usr/bin/env python3
"""Prepare external model/structure assets without coupling them to the core runtime.

The baseline ensemble is trained from GB1 labels and therefore has no pretrained weights to
download. The optional structure profile downloads PDB 5LDE; future structure scorers consume
its path through the StructureEvidenceProvider interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fitness_agents.config import project_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Download model-adjacent assets")
    parser.add_argument("--profile", choices=["baseline", "structure", "all"], default="baseline")
    parser.add_argument("--output-dir", type=Path, default=root / "models")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "baseline": {
            "external_weights": False,
            "reason": "Ridge and ExtraTrees ensemble is fitted locally from visible labels",
        },
        "excluded": ["EVOLVEpro PLM+RF"],
        "assets": [],
    }
    if args.profile in {"structure", "all"}:
        target = args.output_dir / "5LDE.pdb"
        urllib.request.urlretrieve("https://files.rcsb.org/download/5LDE.pdb", target)
        manifest["assets"].append(
            {
                "name": "GB1 reference structure",
                "path": str(target),
                "source": "https://www.rcsb.org/structure/5LDE",
                "sha256": sha256(target),
            }
        )
    manifest_path = args.output_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()

