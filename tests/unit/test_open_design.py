from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from fitness_agents.config import (
    ActiveLearningConfig,
    CalibratedPosteriorConfig,
    CriticConfig,
    DesignerConfig,
    GenerationConfig,
    HybridBatchAcquisitionConfig,
    KnowledgeConfig,
)
from fitness_agents.contracts.schemas import FitnessObservation, Prediction, Variant
from fitness_agents.data import load_open_design_initial_bundle
from fitness_agents.loop.open_design import OpenDesignRunner
from fitness_agents.mutation import (
    create_open_design_proposer,
    normalize_visible_variants,
    resolve_design_space,
)
from fitness_agents.protein_features import ProteinTaskContext
from fitness_agents.validation import OpenDesignHardValidator, build_draft_batch


def _context() -> ProteinTaskContext:
    source = ProteinTaskContext.from_task(
        SimpleNamespace(
            task_id="tiny",
            protein_id="tiny",
            assay_id="tiny",
            wild_type_sites="C",
            mutable_positions=[2],
            reference_sequence="ACD",
            reference_sequence_path=None,
            sequence_position_offset=1,
            numbering_scheme="one_based",
            assay_conditions={},
            structure_resources=(),
        )
    )
    return source.for_open_design()


def test_all_position_proposer_enumerates_every_non_wild_type_substitution() -> None:
    context = _context()
    config = DesignerConfig(space="open_design", position_policy="all")

    design_space = resolve_design_space(context, config)
    proposals = create_open_design_proposer(config, context, design_space).propose()

    assert len(proposals) == 3 * 19
    assert {item.edits[0].position for item in proposals} == {1, 2, 3}
    assert all(len(item.sequence) == 3 for item in proposals)
    assert all(item.sequence != "ACD" for item in proposals)
    assert len({item.proposal_id for item in proposals}) == len(proposals)


def test_visible_compact_codes_are_projected_onto_the_reference() -> None:
    source = ProteinTaskContext.from_task(
        SimpleNamespace(
            task_id="tiny",
            protein_id="tiny",
            assay_id="tiny",
            wild_type_sites="CD",
            mutable_positions=[2, 3],
            reference_sequence="ACDE",
            reference_sequence_path=None,
            sequence_position_offset=1,
            numbering_scheme="one_based",
            assay_conditions={},
            structure_resources=(),
        )
    )
    variant = Variant("v1", "WE", "WE", "C2W;D3E", 2, "initial_observed")
    observation = FitnessObservation("v1", 1.0, "initial_observed", 0)

    variants, observations = normalize_visible_variants(
        [variant],
        [observation],
        source_context=source,
        open_context=source.for_open_design(),
    )

    assert variants[0].variant == "AWEE"
    assert variants[0].sequence == "AWEE"
    assert variants[0].variant_id == "v1"
    assert observations == [observation]


def test_open_design_runner_outputs_sequences_outside_configured_sites(
    experiment_config, tmp_path
) -> None:
    task = replace(
        experiment_config.task,
        reference_sequence="MVDGVT",
        mutable_positions=[2, 3, 4, 5],
        wild_type_sites="VDGV",
    )
    model = replace(
        experiment_config.model,
        feature_provider="full_sequence_onehot",
        ridge_members=2,
        extra_trees_estimators=12,
    )
    active = ActiveLearningConfig(
        enabled=True,
        posterior=CalibratedPosteriorConfig(
            predictor_models=(model,),
            min_training_size=4,
            min_calibration_size=2,
        ),
        acquisition=HybridBatchAcquisitionConfig(
            exploitation_fraction=0.5,
            exploration_fraction=0.5,
            knowledge_fraction=0.0,
        ),
    )
    config = replace(
        experiment_config,
        rounds=1,
        budget_per_round=4,
        candidate_limit=0,
        task=task,
        model=model,
        output_root=tmp_path / "runs",
        knowledge=KnowledgeConfig(
            physchem=False,
            conservation=False,
            structure=False,
            kg=False,
        ),
        knowledge_enabled=False,
        generation=GenerationConfig(selection_driver="active_learning"),
        active_learning=active,
        designer=DesignerConfig(space="open_design", position_policy="all"),
    )

    summary = OpenDesignRunner(config).run()

    assert summary["candidate_pool_consulted"] is False
    assert summary["open_position_count"] == 6
    run_dir = tmp_path / "runs" / summary["run_id"]
    selected = json.loads((run_dir / "selected_candidates.json").read_text())
    ranked = (run_dir / "ranked_candidates.csv").read_text()
    assert selected and all(len(item["sequence"]) == 6 for item in selected)
    assert "M1A" in ranked or "T6A" in ranked
    hypothesis = json.loads((run_dir / "hypothesis.json").read_text())
    assert 1 <= len(hypothesis["preferred_residues"]) <= 12


def test_open_design_rejects_legacy_candidate_prefilter(experiment_config) -> None:
    with pytest.raises(ValueError, match="candidate_limit=0"):
        replace(
            experiment_config,
            task=replace(
                experiment_config.task,
                reference_sequence="MVDGVT",
                mutable_positions=[2, 3, 4, 5],
            ),
            generation=GenerationConfig(selection_driver="active_learning"),
            active_learning=ActiveLearningConfig(enabled=True),
            designer=DesignerConfig(space="open_design", position_policy="all"),
        )


def test_open_design_rejects_gb1_compact_predictor_at_config_time(
    experiment_config,
) -> None:
    task = replace(
        experiment_config.task,
        reference_sequence="MVDGVT",
        mutable_positions=[2, 3, 4, 5],
        wild_type_sites="VDGV",
    )
    compact_model = replace(
        experiment_config.model,
        feature_provider="gb1_onehot_pairwise",
    )
    active = ActiveLearningConfig(
        enabled=True,
        posterior=CalibratedPosteriorConfig(predictor_models=(compact_model,)),
    )

    with pytest.raises(ValueError, match="GB1 four-site predictors"):
        replace(
            experiment_config,
            candidate_limit=0,
            task=task,
            model=compact_model,
            generation=GenerationConfig(selection_driver="active_learning"),
            active_learning=active,
            designer=DesignerConfig(space="open_design", position_policy="all"),
        )


def test_standalone_initial_measurements_need_no_candidate_pool(tmp_path) -> None:
    path = tmp_path / "initial.csv"
    pd.DataFrame(
        [
            {"variant_id": "v0", "variant": "ACD", "fitness": 0.1},
            {"variant_id": "v1", "variant": "AWD", "fitness": 0.4},
        ]
    ).to_csv(path, index=False)

    bundle = load_open_design_initial_bundle(initial_path=path)

    assert bundle.source == "standalone_initial_observations"
    assert [item.variant_id for item in bundle.variants] == ["v0", "v1"]
    assert [item.fitness for item in bundle.observations] == [0.1, 0.4]
    assert not hasattr(bundle, "oracle_pool")


@pytest.mark.parametrize(
    ("sequence", "notation", "expected_code"),
    [
        ("WCD", "A1W", "FORBIDDEN_POSITION"),
        ("AWD", "C2F", "MUTATION_NOTATION_MISMATCH"),
    ],
)
def test_open_design_hard_validator_rejects_position_or_notation_tampering(
    sequence: str,
    notation: str,
    expected_code: str,
) -> None:
    context = _context()
    designer = DesignerConfig(
        space="open_design",
        position_policy="include",
        include_positions=(2,),
    )
    design_space = resolve_design_space(context, designer)
    variant_id = f"sha256:{hashlib.sha256(sequence.encode('ascii')).hexdigest()}"
    variant = Variant(
        variant_id,
        sequence,
        sequence,
        notation,
        1,
        "open_design_candidate",
    )
    prediction = Prediction(variant_id, 0.2, 0.1, (0.0, 0.4), 0.0, {}, "test")
    draft = build_draft_batch(
        round_id=1,
        review_attempt=0,
        candidate_ids=(variant_id,),
        variants={variant_id: variant},
        predictions={variant_id: prediction},
        evidence={},
        hypothesis_id=None,
        falsification_spec=None,
    )
    report = OpenDesignHardValidator(
        design_space,
        CriticConfig(),
        mutation_depth=1,
    ).validate(
        draft,
        variants={variant_id: variant},
        predictions={variant_id: prediction},
        evidence={},
        revealed_ids=set(),
        pending_ids=set(),
        allowed_ids={variant_id},
        expected_batch_size=1,
    )

    assert expected_code in {item.code for item in report.hard_conflicts}
