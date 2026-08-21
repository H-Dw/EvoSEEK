"""Validate the standalone Attachment D results-showcase outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from build_results_showcase import (
    FIGURE_DIR,
    MANIFEST_PATH,
    PROMPT_CHAIN_PATH,
    REPORT_PATH,
    SOURCE_DATA_DIR,
)


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    condition = manifest["selected_condition"]
    fold = int(manifest["selected_fold"])
    top_k_text = manifest["selection_rule"]["top_k"]
    top_k = int(top_k_text.split()[1])
    top_path = SOURCE_DATA_DIR / f"showcase_top{top_k}_by_round.csv"
    top = pd.read_csv(top_path)
    counts = top.groupby("round_id").size().to_dict()
    assert counts == {1: top_k, 2: top_k, 3: top_k}, counts
    assert top["wet_fitness"].notna().all()
    assert top["selection_arm"].ne("not_recorded").all()

    trajectory = pd.read_csv(SOURCE_DATA_DIR / "showcase_fitness_trajectory.csv")
    assert trajectory["round_id"].tolist() == [0, 1, 2, 3]
    assert trajectory["best_seen_fitness"].is_monotonic_increasing
    assert trajectory.iloc[-1]["best_seen_gain_from_initial"] > 0

    sequences = pd.read_csv(SOURCE_DATA_DIR / "showcase_sequences.csv")
    assert sequences.iloc[0]["mutation_notation"] == "WT"
    assert sequences.iloc[0]["variant"] == "VDGV"
    assert len(sequences) >= 4
    assert sequences["full_sequence"].str.len().nunique() == 1

    prompt = json.loads(PROMPT_CHAIN_PATH.read_text(encoding="utf-8"))
    assert prompt["hidden_reasoning_exported"] is False
    assert prompt["draft_and_revision"]["attempt_0"]["critic_verdict"] == "REVISE"
    assert prompt["draft_and_revision"]["attempt_1"]["critic_verdict"] == "APPROVE"
    assert prompt["batch_critic"]["verdict"] == "APPROVE"
    assert prompt["post_measurement_assessment"]["status"] == "SUPPORTED"

    report = REPORT_PATH.read_text(encoding="utf-8")
    for required in (
        "# 附件 D｜结果展示（独立模块）",
        "## D1｜野生型与代表突变序列",
        f"## D2｜每轮推荐 Top-{top_k}",
        "## D3｜Fitness 是否提升",
        "## D4｜Agent 的可审计推理过程",
        "## D5｜失败推荐与可能原因",
        "VWAA",
        "LYWC",
        "MAKE_FALSIFIABLE",
    ):
        assert required in report, required
    assert '"reasoning_content":' not in report
    assert "{_doc_relative" not in report
    assert (FIGURE_DIR / f"{condition}_fold{fold}_fitness_trajectory.png").exists()
    print(
        f"Validated Attachment D: {condition}, fold {fold}, "
        f"Top-{top_k}, {len(top)} displayed recommendations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
