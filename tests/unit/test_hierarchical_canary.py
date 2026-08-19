from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from fitness_agents.config import project_root


def _load_canary():
    path = project_root() / "scripts/run_hierarchical_canary.py"
    spec = importlib.util.spec_from_file_location("fitness_agents_hierarchical_canary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_canary_command_is_exactly_one_fold_one_round_four_conditions(tmp_path: Path) -> None:
    module = _load_canary()
    command = module.build_command(
        argparse.Namespace(
            config=Path("config.yaml"),
            fold=2,
            seed=17,
            max_parallel=4,
            timeout_seconds=30.0,
            output_dir=tmp_path / "canary",
            dry_run=True,
        )
    )
    assert command[command.index("--folds") + 1] == "2"
    assert command[command.index("--rounds") + 1] == "1"
    assert command[command.index("--conditions") + 1].split(",") == [
        "kg_base",
        "kg_base_rag",
        "kg_base_al",
        "kg_3features_rag",
    ]
    assert "--dry-run" in command
    assert "--placeholder-predictor" in command
