"""Fail-closed audit for campaign completion manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fitness_agents.contracts.hypothesis_pipeline import CompletionManifest


def audit_run(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    manifest_path = target / "completion_manifest.json" if target.is_dir() else target
    if not manifest_path.is_file():
        return {
            "path": str(manifest_path),
            "passed": False,
            "errors": ["completion_manifest.json is missing"],
        }
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = CompletionManifest.model_validate(raw)
    except (OSError, ValueError) as error:
        return {
            "path": str(manifest_path),
            "passed": False,
            "errors": [f"invalid completion manifest: {type(error).__name__}: {error}"],
        }
    errors: list[str] = []
    if not manifest.pass_eligible:
        errors.append("run is not eligible for evaluation pass")
    if manifest.evaluation_status == "passed" and not manifest.pass_eligible:
        errors.append("false pass: evaluation_status=passed on an ineligible run")
    return {
        "path": str(manifest_path),
        "passed": not errors,
        "errors": errors,
        "manifest": manifest.model_dump(mode="json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Run directories or completion manifests")
    args = parser.parse_args()
    results = [audit_run(path) for path in args.paths]
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
