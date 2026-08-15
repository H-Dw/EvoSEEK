from __future__ import annotations

import json
from dataclasses import replace

from common import (
    ensure,
    load_config,
    parse_args,
    resolve_output,
    write_legacy_benchmark,
    write_result,
)

from fitness_agents.config import (
    CriticConfig,
    ExperimentConfig,
    KnowledgeConfig,
    ModelConfig,
    TaskConfig,
)
from fitness_agents.evaluation import ScientificThinkingEvaluator
from fitness_agents.loop import run_campaign
from fitness_agents.reporting import aggregate_runs, write_science_markdown


def main() -> None:
    args = parse_args("configs/module_tests/campaign_evaluation.yaml")
    config = load_config(args.config)
    output = resolve_output(config, args.output_dir)
    paths = write_legacy_benchmark(output / "input", seed=int(config["seed"]))
    task = TaskConfig(
        task_id="module_campaign",
        protein_id="GB1",
        assay_id="module_test_assay",
        wild_type_sites="VDGV",
        mutable_positions=[39, 40, 41, 54],
        objective="maximize",
        public_data_path=paths["public"],
        oracle_data_path=paths["oracle"],
    )
    model = ModelConfig(**config["model"])
    knowledge_raw = dict(config["knowledge"])
    knowledge_raw["site_profiles"] = {
        int(position): profile
        for position, profile in knowledge_raw.get("site_profiles", {}).items()
    }
    knowledge = KnowledgeConfig(**knowledge_raw)
    critic = CriticConfig(**config["critic"])

    def make_experiment(
        mode: str,
        acquisition: str,
        knowledge_enabled: bool,
        label: str,
        *,
        score_shuffle: bool = False,
        evidence_deletion: bool = False,
    ) -> ExperimentConfig:
        return ExperimentConfig(
            mode=mode,
            seed=int(config["seed"]),
            rounds=int(config["rounds"]),
            budget_per_round=int(config["budget_per_round"]),
            candidate_limit=int(config["candidate_limit"]),
            acquisition=acquisition,
            ucb_beta=float(config["ucb_beta"]),
            diversity_lambda=float(config["diversity_lambda"]),
            task=task,
            model=model,
            knowledge=knowledge,
            critic=critic,
            output_root=output / "runs",
            llm_provider="mock",
            knowledge_enabled=knowledge_enabled,
            score_shuffle=score_shuffle,
            evidence_deletion=evidence_deletion,
            run_label=label,
            evidence_prefilter_limit=200,
        )

    summaries: dict[str, dict] = {}
    baseline_configs: dict[str, ExperimentConfig] = {}
    for item in config["modes"]:
        name = str(item["name"])
        experiment = make_experiment(
            name,
            str(item["acquisition"]),
            bool(item["knowledge_enabled"]),
            f"module-{name}",
        )
        baseline_configs[name] = experiment
        summary = run_campaign(experiment)
        ensure(summary["finalized"], f"{name} campaign did not finalize")
        ensure(
            summary["queries_used"] == int(config["budget_per_round"]) * int(config["rounds"]),
            f"{name} campaign query accounting is wrong",
        )
        run_dir = experiment.output_root / summary["run_id"]
        trace = [
            json.loads(line)
            for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        event_types = [item["event_type"] for item in trace]
        ensure(
            event_types.index("batch_approved") < event_types.index("batch_measured"),
            f"{name} measured a batch before approval",
        )
        ensure((run_dir / "round_01" / "approved_batch.json").is_file(), "Approval artifact missing")
        summaries[name] = summary

    aggregate_paths = aggregate_runs(list(summaries.values()), output / "aggregate")
    ensure(all(path.is_file() for path in aggregate_paths.values()), "Aggregate reports are missing")

    reference_config = baseline_configs["knowledge_agent"]
    intervention_configs = {
        "knowledge_ablation": replace(
            reference_config,
            knowledge_enabled=False,
            run_label="module-science-no-knowledge",
        ),
        "score_shuffle": replace(
            reference_config,
            score_shuffle=True,
            run_label="module-science-score-shuffle",
        ),
        "evidence_deletion": replace(
            reference_config,
            evidence_deletion=True,
            run_label="module-science-delete-evidence",
        ),
    }
    intervention_summaries = {
        name: run_campaign(experiment)
        for name, experiment in intervention_configs.items()
    }
    science_report = ScientificThinkingEvaluator().evaluate(
        reference_dir=summaries["knowledge_agent"]["run_dir"],
        knowledge_ablation_dir=intervention_summaries["knowledge_ablation"]["run_dir"],
        score_shuffle_dir=intervention_summaries["score_shuffle"]["run_dir"],
        evidence_deletion_dir=intervention_summaries["evidence_deletion"]["run_dir"],
    )
    required_metrics = {
        "knowledge_ablation_selection_change",
        "score_shuffle_selection_change",
        "evidence_deletion_selection_change",
        "global_rank_tracking_completeness",
    }
    ensure(required_metrics.issubset(science_report["metrics"]), "Science metrics are incomplete")
    ensure(
        science_report["metrics"]["global_rank_tracking_completeness"] == 1.0,
        "Global candidate ranks were not fully tracked",
    )
    science_path = write_science_markdown(science_report, output / "scientific_thinking.md")
    ensure(science_path.is_file(), "Scientific-thinking Markdown was not written")

    write_result(
        output,
        "campaign_evaluation",
        {
            "config": config["_config_path"],
            "baseline_summaries": summaries,
            "aggregate_reports": aggregate_paths,
            "intervention_run_ids": {
                name: summary["run_id"] for name, summary in intervention_summaries.items()
            },
            "scientific_thinking": science_report,
            "scientific_thinking_markdown": science_path,
        },
    )


if __name__ == "__main__":
    main()

