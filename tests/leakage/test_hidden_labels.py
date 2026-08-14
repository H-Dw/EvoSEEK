import json

import pandas as pd
import pytest

from fitness_agents.agents.scientist import assert_sanitized
from fitness_agents.data.loader import load_dataset_bundle
from fitness_agents.loop import CsvOracleBackend, run_campaign


@pytest.mark.leakage
def test_public_loader_rejects_fitness_column(synthetic_benchmark, tmp_path):
    public = pd.read_csv(synthetic_benchmark["public"])
    public["fitness"] = 1.0
    bad_path = tmp_path / "bad_public.csv"
    public.to_csv(bad_path, index=False)
    with pytest.raises(ValueError, match="hidden labels"):
        load_dataset_bundle(bad_path, synthetic_benchmark["oracle"])


@pytest.mark.leakage
def test_oracle_rejects_repeat_and_post_final_submission(synthetic_benchmark):
    backend = CsvOracleBackend(synthetic_benchmark["oracle"])
    labels = pd.read_csv(synthetic_benchmark["oracle"])
    variant_id = labels.loc[labels["split_role"] == "oracle_pool", "variant_id"].iloc[0]
    run_id = backend.submit([variant_id], 1)
    backend.collect(run_id)
    with pytest.raises(PermissionError):
        backend.submit([variant_id], 2)
    backend.open_final_test()
    with pytest.raises(RuntimeError):
        backend.submit([], 3)


@pytest.mark.leakage
def test_prompt_sanitizer_rejects_final_and_oracle_keys():
    with pytest.raises(ValueError):
        assert_sanitized({"nested": {"final_test": ["secret"]}})
    with pytest.raises(ValueError):
        assert_sanitized({"oracle_data_path": "/secret/oracle.csv"})


@pytest.mark.leakage
def test_unselected_hidden_label_never_appears_in_trace(config_factory):
    config = config_factory(rounds=1, budget_per_round=2, run_label="leakage")
    summary = run_campaign(config)
    trace = (config.output_root / summary["run_id"] / "trace.jsonl").read_text()
    assert "98765.4321" not in trace
    config_record = json.loads(
        (config.output_root / summary["run_id"] / "config.json").read_text()
    )
    assert "oracle_data_path" not in config_record

