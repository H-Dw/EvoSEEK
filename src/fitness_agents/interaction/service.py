"""UI-independent application service for open sequence design."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from fitness_agents.agents.critic import CRITIQUE_DECISION_SCHEMA, RuleBasedCriticClient
from fitness_agents.agents.remote_llm import resolve_api_key
from fitness_agents.config import ExperimentConfig, load_experiment_config
from fitness_agents.contracts.design import ResolvedDesignSpace
from fitness_agents.contracts.interaction import (
    EvolutionIntent,
    EvolutionRunResult,
    OpenDesignRequestPreview,
)
from fitness_agents.contracts.schemas import ConflictReport, Evidence, Prediction, Variant
from fitness_agents.loop.open_design import OpenDesignRunner
from fitness_agents.models.capabilities import predictor_capabilities
from fitness_agents.mutation import resolve_design_space
from fitness_agents.protein_features import ProteinTaskContext
from fitness_agents.validation.batch import CritiqueDecisionValidator, build_draft_batch

from .intent import DeterministicEvolutionIntentParser


@dataclass(frozen=True)
class _PreparedRequest:
    config: ExperimentConfig
    design_space: ResolvedDesignSpace
    preview: OpenDesignRequestPreview


class OpenDesignRunError(RuntimeError):
    """Runner failure that carries the run directory for UI diagnostics."""

    def __init__(self, error: BaseException, *, run_dir: Path) -> None:
        super().__init__(f"{type(error).__name__}: {error}")
        self.run_dir = str(run_dir)


class EvolutionApplicationService:
    """Compile a bounded prompt, preview it, then run only after confirmation."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.base_config = load_experiment_config(self.config_path)
        if self.base_config.designer.space != "open_design":
            raise ValueError("Interaction service requires a trusted open_design config")
        self.parser = DeterministicEvolutionIntentParser()
        self._prepared: dict[str, _PreparedRequest] = {}
        self._preview_seq = 0
        self.preflight_report = self._preflight()

    def _preflight(self) -> tuple[str, ...]:
        """Fail-closed startup checks mirroring the batch runner's preflight."""

        checks: list[str] = []
        failures: list[str] = []
        config = self.base_config
        posterior_models = (
            config.active_learning.posterior.predictor_models or (config.model,)
        )
        capabilities = [predictor_capabilities(item) for item in posterior_models]
        if all(item.supports_full_sequence for item in capabilities) and all(
            item.supports_generated_sequences for item in capabilities
        ):
            checks.append("posterior predictor 支持完整生成序列")
        else:
            failures.append("配置的 posterior predictor 不支持完整生成序列。")

        smoke_error = self._critic_smoke_check(config)
        if smoke_error is None:
            checks.append("规则 Critic 冒烟决策通过 schema 与确定性校验")
        else:
            failures.append(f"规则 Critic 冒烟测试失败：{smoke_error}")

        if config.critic.mode == "remote" and config.critic.enabled:
            try:
                resolve_api_key(config.critic.api_key)
            except Exception as error:  # noqa: BLE001 - preflight aggregates failures
                failures.append(f"remote critic API key 不可解析：{error}")
            else:
                checks.append("remote critic API key 可解析")
            if isinstance(config.critic.max_tokens, int) and config.critic.max_tokens > 0:
                checks.append(f"critic max_tokens 已封顶为 {config.critic.max_tokens}")
            else:
                failures.append("remote critic 需要正整数 max_tokens 预算。")
        if failures:
            raise ValueError("交互服务启动预检失败：" + "；".join(failures))
        return tuple(checks)

    @staticmethod
    def _critic_smoke_check(config: ExperimentConfig) -> str | None:
        """Run the deterministic critic over a synthetic full-size batch."""

        candidate_ids = tuple(f"smoke:{index}" for index in range(config.budget_per_round))
        variants = {
            item: Variant(
                variant_id=item,
                variant=item,
                sequence="A",
                mutation_notation="WT",
                mutation_count=0,
                split_role="generated",
            )
            for item in candidate_ids
        }
        predictions = {
            item: Prediction(
                variant_id=item,
                fitness_mean=0.0,
                fitness_std=1.0,
                interval_90=(-1.0, 1.0),
                ood_score=0.0,
                component_scores={},
                model_version="smoke",
            )
            for item in candidate_ids
        }
        channels = ("physchem", "conservation", "structure", "kg")
        evidence = {
            item: [
                Evidence(
                    evidence_id=f"smoke:{channel}:{item}",
                    variant_id=item,
                    channel=channel,
                    statement="smoke",
                    score=0.0,
                    source_id="smoke",
                    confidence=0.0,
                    round_id=1,
                )
                for channel in channels
            ]
            for item in candidate_ids
        }
        draft = build_draft_batch(
            round_id=1,
            review_attempt=0,
            candidate_ids=candidate_ids,
            variants=variants,
            predictions=predictions,
            evidence=evidence,
            hypothesis_id=None,
            falsification_spec=None,
        )
        report = ConflictReport(
            report_id="smoke",
            round_id=1,
            conflicts=(),
            validator_version="smoke",
            draft_batch_id=draft.draft_batch_id,
        )
        try:
            decision = RuleBasedCriticClient().review(
                context={"draft": draft, "conflict_report": report, "evidence": evidence},
                output_schema=CRITIQUE_DECISION_SCHEMA,
            )
            CritiqueDecisionValidator().validate(
                decision,
                draft=draft,
                report=report,
                visible_evidence_ids={
                    entry.evidence_id for entries in evidence.values() for entry in entries
                },
            )
        except Exception as error:  # noqa: BLE001 - preflight reports, never raises raw
            return f"{type(error).__name__}: {error}"
        return None

    @property
    def configured_reference(self) -> str:
        return ProteinTaskContext.from_task(self.base_config.task).full_sequence

    def interpret(
        self, prompt: str, *, sequence_text: str | None = None
    ) -> EvolutionIntent:
        return self.parser.parse(
            prompt,
            sequence_text=sequence_text,
            configured_reference=self.configured_reference,
            configured_objective=self.base_config.task.objective,
        )

    def preview(
        self, prompt: str, *, sequence_text: str | None = None
    ) -> OpenDesignRequestPreview:
        return self.preview_intent(self.interpret(prompt, sequence_text=sequence_text))

    def preview_intent(self, intent: EvolutionIntent) -> OpenDesignRequestPreview:
        blockers = list(intent.missing_fields)
        warnings: list[str] = []
        reference = intent.reference_sequence or ""
        configured_reference = self.configured_reference
        if reference and reference != configured_reference:
            blockers.append(
                "输入序列与可信配置的 reference sequence 不一致；新蛋白必须使用匹配的任务配置和初始测量。"
            )
        if intent.requested_depth != 1:
            blockers.append("当前 open_design proposer 仅支持精确单点替换（mutation_depth=1）。")
        if intent.requested_rounds != 1:
            blockers.append("当前开放设计路径是单轮 design/export；多轮需要 sequence-aware measurement backend。")

        budget = intent.requested_budget or self.base_config.budget_per_round
        if budget > self.base_config.budget_per_round:
            warnings.append(
                f"请求预算 {budget} 超过可信配置上限，已限制为 {self.base_config.budget_per_round}。"
            )
            budget = self.base_config.budget_per_round
        objective = intent.objective_text or self.base_config.task.objective
        designer = replace(
            self.base_config.designer,
            position_policy=intent.constraints.position_policy,
            include_positions=intent.constraints.include_positions,
            exclude_positions=intent.constraints.exclude_positions,
            mutation_depth=intent.requested_depth or 1,
        )
        config: ExperimentConfig | None = None
        design_space: ResolvedDesignSpace | None = None
        try:
            config = replace(
                self.base_config,
                budget_per_round=budget,
                task=replace(self.base_config.task, objective=objective),
                designer=designer,
                run_label="interactive-open-design",
            )
            computation_context = ProteinTaskContext.from_task(config.task).for_open_design()
            design_space = resolve_design_space(computation_context, config.designer)
        except ValueError as error:
            blockers.append(str(error))

        posterior_models = (
            self.base_config.active_learning.posterior.predictor_models
            or (self.base_config.model,)
        )
        capabilities = [predictor_capabilities(item) for item in posterior_models]
        supports_full = all(item.supports_full_sequence for item in capabilities)
        supports_generated = all(
            item.supports_generated_sequences for item in capabilities
        )
        if not supports_full or not supports_generated:
            blockers.append("配置的 posterior predictor 不支持完整生成序列。")

        positions = (
            design_space.allowed_mutation_positions if design_space is not None else ()
        )
        generated_count = 0
        if design_space is not None:
            generated_count = sum(
                sum(
                    residue != design_space.residue_at(position)
                    for residue in design_space.allowed_residues
                )
                for position in positions
            )
        self._preview_seq += 1
        preview_id = f"PV{self._preview_seq:02d}"
        ready = not blockers and config is not None and design_space is not None
        preview = OpenDesignRequestPreview(
            preview_id=preview_id,
            objective_text=objective,
            reference_id=intent.reference_id or "REF01",
            reference_length=len(reference or configured_reference),
            position_policy=intent.constraints.position_policy,
            resolved_positions=positions,
            resolved_position_count=len(positions),
            mutation_depth=intent.requested_depth or 1,
            budget=budget,
            generated_candidate_count=generated_count,
            supports_full_sequence=supports_full,
            supports_generated_sequences=supports_generated,
            initial_data_source=(
                "initial_observations"
                if self.base_config.task.initial_observations_path is not None
                else "configured_visible_fold"
            ),
            ready_for_confirmation=ready,
            blockers=tuple(dict.fromkeys(blockers)),
            warnings=tuple(warnings),
            confirmation_summary=intent.confirmation_summary,
        )
        if ready and config is not None and design_space is not None:
            self._prepared[preview_id] = _PreparedRequest(config, design_space, preview)
        return preview

    def run(self, preview_id: str, *, confirmed: bool) -> EvolutionRunResult:
        if not confirmed:
            raise PermissionError("必须确认结构化 preview 后才能创建运行任务。")
        try:
            prepared = self._prepared[preview_id]
        except KeyError as error:
            raise ValueError("preview 不存在、未通过预检，或已经提交。") from error
        runner = OpenDesignRunner(
            prepared.config,
            resolved_design_space=prepared.design_space,
        )
        try:
            summary = runner.run()
        except Exception as error:
            # Keep the prepared request so the user can adjust inputs and retry.
            raise OpenDesignRunError(error, run_dir=runner.writer.run_dir) from error
        del self._prepared[preview_id]
        run_dir = Path(str(summary["run_dir"])).resolve()
        allowed_names = (
            "selected_candidates.fasta",
            "selected_candidates.csv",
            "selected_candidates.json",
            "summary.json",
            "review/approved_batch.json",
        )
        artifacts = tuple(
            str(path)
            for name in allowed_names
            if (path := (run_dir / name).resolve()).is_file()
            and run_dir in path.parents
        )
        return EvolutionRunResult(
            preview_id=preview_id,
            status="completed",
            run_id=str(summary["run_id"]),
            public_message=(
                "开放设计完成；最终候选已通过 hard validation、Critic 和 Approval。"
                f"\nrun_id: {summary['run_id']}\nrun_dir: {run_dir}"
            ),
            summary=summary,
            artifact_paths=artifacts,
        )
