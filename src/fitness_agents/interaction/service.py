"""UI-independent application service for open sequence design."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from fitness_agents.config import ExperimentConfig, load_experiment_config
from fitness_agents.contracts.design import ResolvedDesignSpace
from fitness_agents.contracts.interaction import (
    EvolutionIntent,
    EvolutionRunResult,
    OpenDesignRequestPreview,
)
from fitness_agents.loop.open_design import OpenDesignRunner
from fitness_agents.models.capabilities import predictor_capabilities
from fitness_agents.mutation import resolve_design_space
from fitness_agents.protein_features import ProteinTaskContext

from .intent import DeterministicEvolutionIntentParser


@dataclass(frozen=True)
class _PreparedRequest:
    config: ExperimentConfig
    design_space: ResolvedDesignSpace
    preview: OpenDesignRequestPreview


class EvolutionApplicationService:
    """Compile a bounded prompt, preview it, then run only after confirmation."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.base_config = load_experiment_config(self.config_path)
        if self.base_config.designer.space != "open_design":
            raise ValueError("Interaction service requires a trusted open_design config")
        self.parser = DeterministicEvolutionIntentParser()
        self._prepared: dict[str, _PreparedRequest] = {}

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
        payload = {
            "intent": intent.model_dump(mode="json"),
            "config": str(self.config_path.resolve()),
            "positions": positions,
            "budget": budget,
            "reference_sha256": intent.reference_sequence_sha256,
        }
        preview_id = "preview:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:20]
        ready = not blockers and config is not None and design_space is not None
        preview = OpenDesignRequestPreview(
            preview_id=preview_id,
            objective_text=objective,
            reference_sequence_sha256=(
                intent.reference_sequence_sha256
                or hashlib.sha256(configured_reference.encode("ascii")).hexdigest()
            ),
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
            prepared = self._prepared.pop(preview_id)
        except KeyError as error:
            raise ValueError("preview 不存在、未通过预检，或已经提交。") from error
        summary = OpenDesignRunner(
            prepared.config,
            resolved_design_space=prepared.design_space,
        ).run()
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
            public_message="开放设计完成；最终候选已通过 hard validation、Critic 和 Approval。",
            summary=summary,
            artifact_paths=artifacts,
        )
