from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from fitness_agents.agents.remote_llm import load_project_env, resolve_secret


pytestmark = pytest.mark.live_llm


def _load_live_module():
    module_dir = Path(__file__).resolve().parents[2] / "scripts" / "module_tests"
    sys.path.insert(0, str(module_dir))
    path = module_dir / "test_llm_output_stability.py"
    spec = importlib.util.spec_from_file_location("llm_output_stability_live", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_llm_output_stability_replay() -> None:
    load_project_env()
    if os.environ.get("FITNESS_AGENTS_LIVE_LLM") != "1":
        pytest.skip("Set FITNESS_AGENTS_LIVE_LLM=1 to call the real LLM API")
    if not resolve_secret(None, "DEEPSEEK_API_KEY", "FITNESS_AGENTS_LLM_API_KEY", "OPENAI_API_KEY"):
        pytest.skip("No LLM API key in the environment")

    live = _load_live_module()
    config = live.load_config("configs/module_tests/agents_review.yaml")
    remote = live._configure_remote(dict(config.get("remote_llm") or {}))
    replay = live._run_hypothesis_replay(remote, repeats=1)
    revise = live._run_revise_reproposal(remote)
    assert replay["n"] >= 1
    assert replay["guarded_contract_ok"] + replay["schema_ok"] >= 1
    assert revise["parent_matches"] is True
    assert revise["unknown_evidence_ids"] == []
