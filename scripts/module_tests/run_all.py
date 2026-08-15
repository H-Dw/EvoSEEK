from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import REPO_ROOT, jsonable

MODULES = (
    ("data_pipeline", "test_data_pipeline.py", "data_pipeline.yaml"),
    ("predictive_model", "test_predictive_model.py", "predictive_model.yaml"),
    ("design_acquisition", "test_design_acquisition.py", "design_acquisition.yaml"),
    ("knowledge_runtime", "test_knowledge_runtime.py", "knowledge_runtime.yaml"),
    ("kg_construction", "test_kg_construction.py", "kg_construction.yaml"),
    ("kg_interaction", "test_kg_interaction.py", "kg_interaction.yaml"),
    ("agents_review", "test_agents_review.py", "agents_review.yaml"),
    ("campaign_evaluation", "test_campaign_evaluation.py", "campaign_evaluation.yaml"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every standalone module functional test")
    parser.add_argument("--output-root", default="artifacts/module_tests")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for module, script_name, config_name in MODULES:
        command = [
            sys.executable,
            str(Path(__file__).with_name(script_name)),
            "--config",
            str(REPO_ROOT / "configs/module_tests" / config_name),
            "--output-dir",
            str(output_root / module),
        ]
        print(f"[module-test] starting {module}", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        result_path = output_root / module / "result.json"
        record: dict[str, object] = {
            "module": module,
            "returncode": completed.returncode,
            "result_path": result_path,
        }
        if completed.returncode == 0 and result_path.is_file():
            record["result"] = json.loads(result_path.read_text(encoding="utf-8"))
        results.append(record)
        if completed.returncode != 0 and args.stop_on_failure:
            break

    failed = [item for item in results if item["returncode"] != 0]
    summary = {
        "status": "failed" if failed else "passed",
        "modules_requested": len(MODULES),
        "modules_executed": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "results": results,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "summary": str(summary_path)}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

