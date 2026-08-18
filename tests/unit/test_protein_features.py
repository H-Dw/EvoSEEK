from __future__ import annotations

from pathlib import Path

from fitness_agents.config import (
    KnowledgeConfig,
    KnowledgeProviderConfig,
    TaskConfig,
    load_experiment_config,
)
from fitness_agents.contracts.schemas import Evidence, FitnessObservation, Variant
from fitness_agents.knowledge import KnowledgeEngine
from fitness_agents.protein_features import (
    MSAProfileProvider,
    PhyschemDescriptorProvider,
    ProteinTaskContext,
    StaticStructureProvider,
)
from fitness_agents.protein_features.calibration import calibrate_visible_evidence


def _task(tmp_path: Path, **changes) -> TaskConfig:
    values = {
        "task_id": "dynamic-protein",
        "protein_id": "PTEST",
        "assay_id": "assay",
        "wild_type_sites": "CE",
        "mutable_positions": [2, 4],
        "objective": "maximize",
        "public_data_path": tmp_path / "public.csv",
        "oracle_data_path": tmp_path / "oracle.csv",
        "reference_sequence": "ACDE",
        "sequence_position_offset": 1,
        "numbering_scheme": "domain_1_based",
    }
    values.update(changes)
    return TaskConfig(**values)


def _variant(variant_id: str, code: str) -> Variant:
    return Variant(
        variant_id=variant_id,
        variant=code,
        sequence=code,
        mutation_notation=code,
        mutation_count=0,
        split_role="candidate",
    )


def test_task_context_maps_non_gb1_positions_and_builds_full_sequence(tmp_path: Path) -> None:
    context = ProteinTaskContext.from_task(_task(tmp_path))

    assert context.mutable_positions == (2, 4)
    assert context.wild_type_code == "CE"
    assert context.position_to_sequence_index == {2: 1, 4: 3}
    assert context.full_sequence_for_variant("CF") == "ACDF"


def test_physchem_provider_emits_source_backed_descriptor_without_claiming_fitness(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    context = ProteinTaskContext.from_task(_task(tmp_path))
    provider = PhyschemDescriptorProvider(
        context,
        KnowledgeProviderConfig(
            kind="aaindex_delta",
            resource_path=root / "configs/resources/aaindex_minimal.yaml",
        ),
        parameter_set_id="test:v1",
    )

    evidence = provider.evaluate(_variant("v1", "CF"), round_id=1)

    assert evidence.quality_status == "ok"
    assert not evidence.contributes_to_selection
    assert evidence.confidence == 0.0
    assert evidence.raw_features["sites"]["4"]["mutation"] == "E4F"
    assert "not an assay-fitness claim" in evidence.statement
    assert evidence.provenance["resource_sha256"]


def test_msa_profile_is_computed_once_and_content_addressed(tmp_path: Path) -> None:
    alignment = tmp_path / "alignment.a3m"
    alignment.write_text(
        ">query\nACDE\n>s2\nACDF\n>s3\nACNE\n>s4\nACDE\n",
        encoding="utf-8",
    )
    context = ProteinTaskContext.from_task(_task(tmp_path))
    config = KnowledgeProviderConfig(
        kind="msa_profile",
        a3m_path=alignment,
        options={
            "identity_threshold": 0.8,
            "pseudocount": 0.5,
            "minimum_neff": 1.0,
            "minimum_sequence_coverage": 0.5,
            "maximum_sequence_gap_fraction": 0.5,
        },
    )

    first = MSAProfileProvider(
        context,
        config,
        parameter_set_id="test:v1",
        cache_dir=tmp_path / "cache",
    )
    second = MSAProfileProvider(
        context,
        config,
        parameter_set_id="test:v1",
        cache_dir=tmp_path / "cache",
    )
    evidence = second.evaluate(_variant("v1", "CF"), round_id=1)

    assert first.cache_status == "miss"
    assert second.cache_status == "hit"
    assert evidence.raw_features["neff"] > 0
    assert evidence.raw_features["sites"]["4"]["mutant_frequency"] > 0
    assert "evolutionary prior, not assay fitness" in evidence.statement
    assert evidence.provenance["input_mode"] == "precomputed_a3m"
    assert evidence.provenance["a3m_path"] == str(alignment)


def test_msa_neff_scaled_uniform_prior_does_not_grow_with_state_space(
    tmp_path: Path,
) -> None:
    alignment = tmp_path / "alignment.a3m"
    alignment.write_text(
        ">query\nACDE\n>s2\nACDF\n>s3\nACNE\n>s4\nATDF\n",
        encoding="utf-8",
    )
    context = ProteinTaskContext.from_task(_task(tmp_path))
    provider = MSAProfileProvider(
        context,
        KnowledgeProviderConfig(
            kind="msa_profile",
            a3m_path=alignment,
            options={
                "identity_threshold": 0.8,
                "pseudocount_mode": "neff_scaled_uniform",
                "pseudocount_weight": 0.25,
                "minimum_single_site_neff": 1.0,
                "minimum_site_effective_count": 1.0,
                "minimum_sequence_coverage": 0.5,
                "maximum_sequence_gap_fraction": 0.5,
                "single_site_aggregation": "sum_log_odds",
                "pairwise_enabled": True,
                "pairwise_mode": "marginal_corrected_log_odds",
                "pairwise_minimum_neff_per_length": 0.0,
                "estimated_parameters": ["pseudocount_weight"],
            },
        ),
        parameter_set_id="test:v2",
        cache_dir=tmp_path / "cache",
    )

    evidence = provider.evaluate(_variant("v1", "CF"), round_id=1)
    expected_total = 0.25 * provider.profile.neff

    assert provider.profile.settings["single_pseudocount_total"] == expected_total
    assert provider.profile.settings["pair_pseudocount_total"] == expected_total
    assert evidence.raw_features["pseudocount_mode"] == "neff_scaled_uniform"
    assert evidence.raw_features["pairwise_eligible"] is True
    assert evidence.raw_features["pairwise_frequency_log_odds"] is None
    assert evidence.raw_features["pairwise_residual_log_odds"] is not None
    assert evidence.raw_features["sites"]["4"]["effective_count"] > 0
    assert "pairwise_residual_not_direct_coupling" in evidence.warnings


def _pdb_atom(serial: int, name: str, residue: str, number: int, x: float) -> str:
    element = name[0]
    return (
        f"ATOM  {serial:5d} {name:^4s} {residue:>3s} A{number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}          {element:>2s}"
    )


def test_static_structure_provider_uses_coordinates_and_reports_limitations(
    tmp_path: Path,
) -> None:
    structure = tmp_path / "test.pdb"
    lines = []
    serial = 1
    for number, residue, origin in ((1, "ALA", 0.0), (2, "CYS", 3.0), (3, "ASP", 6.0)):
        for name, offset in (("N", 0.0), ("CA", 0.8), ("C", 1.6), ("O", 2.1)):
            lines.append(_pdb_atom(serial, name, residue, number, origin + offset))
            serial += 1
    structure.write_text("\n".join(lines) + "\n", encoding="utf-8")
    task = _task(
        tmp_path,
        wild_type_sites="C",
        mutable_positions=[2],
        structure_resources=(
            {
                "resource_id": "test-structure",
                "path": structure,
                "format": "pdb",
                "chain": "A",
            },
        ),
    )
    context = ProteinTaskContext.from_task(task)
    provider = StaticStructureProvider(
        context,
        KnowledgeProviderConfig(
            kind="static_structure",
            options={
                "contact_cutoff_angstrom": 5.0,
                "interface_cutoff_angstrom": 5.0,
                "hbond_cutoff_angstrom": 3.5,
                "salt_bridge_cutoff_angstrom": 4.0,
                "sasa_probe_radius_angstrom": 1.4,
                "sasa_sphere_points": 24,
                "dense_contact_count": 2,
                "clash_distance_fraction": 0.75,
                "disulfide_sg_cutoff_angstrom": 2.3,
            },
        ),
        parameter_set_id="test:v1",
    )

    evidence = provider.evaluate(_variant("v1", "F"), round_id=1)

    site = evidence.raw_features["sites"]["2"]
    assert site["status"] == "ok"
    assert site["contact_count"] >= 1
    assert site["sasa_angstrom2"] >= 0
    assert site["mutant_side_chain_not_modelled"] is True
    assert "no folding or affinity claim" in evidence.statement


def test_visible_calibration_never_uses_unrevealed_labels() -> None:
    config = KnowledgeProviderConfig(
        kind="aaindex_delta",
        contributes_to_selection=True,
        calibration="visible_linear",
        minimum_calibration_samples=2,
    )
    evidence = {
        variant_id: [
            Evidence(
                evidence_id=f"e:{variant_id}",
                variant_id=variant_id,
                channel="physchem",
                statement="descriptor",
                score=score,
                source_id="test",
                confidence=0.0,
                round_id=1,
                quality_status="ok",
            )
        ]
        for variant_id, score in (("visible-a", 0.1), ("visible-b", 0.9), ("hidden", 0.5))
    }
    observations = {
        "visible-a": FitnessObservation(
            "visible-a", 0.0, "initial_observed", 0, "test"
        ),
        "visible-b": FitnessObservation(
            "visible-b", 1.0, "initial_observed", 0, "test"
        ),
    }

    calibrated = calibrate_visible_evidence(
        evidence,
        observations,
        {"physchem": config},
    )

    hidden = calibrated["hidden"][0]
    assert hidden.calibrated
    assert hidden.contributes_to_selection
    assert hidden.provenance["calibration"]["sample_count"] == 2.0
    assert hidden.provenance["calibration"]["label_scope"] == (
        "already_visible_measurements_only"
    )


def test_missing_scientific_resource_is_unavailable_not_neutral_score(
    tmp_path: Path,
) -> None:
    context = ProteinTaskContext.from_task(_task(tmp_path))
    knowledge = KnowledgeConfig(
        physchem=False,
        conservation=True,
        structure=False,
        kg=False,
        legacy_mode=False,
        providers={
            "conservation": KnowledgeProviderConfig(
                kind="msa_profile",
                options={
                    "identity_threshold": 0.8,
                    "pseudocount": 0.5,
                    "minimum_neff": 16.0,
                    "minimum_sequence_coverage": 0.7,
                    "maximum_sequence_gap_fraction": 0.5,
                },
            )
        },
    )
    engine = KnowledgeEngine(
        knowledge,
        graph_path=tmp_path / "kg.sqlite",
        assay_id="assay",
        task_context=context,
    )

    evidence = engine.evidence_for([_variant("v1", "CF")], round_id=1)["v1"][0]

    assert evidence.quality_status == "unavailable"
    assert evidence.score == 0.0
    assert evidence.confidence == 0.0
    assert not evidence.contributes_to_selection
    assert engine.provider_status["conservation"]["status"] == "unavailable"


def test_gb1_example_a3m_and_cif_return_usable_typed_evidence(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    config = load_experiment_config(
        root / "configs/experiments/knowledge_agent_features.example.yaml"
    )
    context = ProteinTaskContext.from_task(config.task)
    engine = KnowledgeEngine(
        config.knowledge,
        graph_path=tmp_path / "gb1.sqlite",
        structured_graph_path=tmp_path / "gb1-structured.sqlite",
        assay_id=config.task.assay_id,
        protein_id=config.task.protein_id,
        task_context=context,
        local_knowledge_enabled=False,
    )
    variant = Variant(
        "gb1-demo",
        "WEGV",
        context.full_sequence_for_variant("WEGV"),
        "V39W;D40E",
        2,
        "candidate",
    )

    evidence = engine.evidence_for([variant], round_id=1)[variant.variant_id]
    by_channel = {item.channel: item for item in evidence}

    assert context.full_sequence == "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    assert by_channel["physchem"].quality_status == "ok"
    assert by_channel["conservation"].quality_status == "ok"
    assert by_channel["conservation"].provenance["input_mode"] == "precomputed_a3m"
    assert by_channel["conservation"].provenance["a3m_path"].endswith("non_pairing.a3m")
    assert by_channel["conservation"].raw_features["sequence_count"] == 39
    assert by_channel["conservation"].raw_features["neff"] > 12.0
    assert by_channel["conservation"].raw_features["neff_per_length"] < 0.3
    assert by_channel["conservation"].raw_features["pairwise_enabled"] is False
    assert by_channel["conservation"].raw_features["pairwise_eligible"] is False
    assert by_channel["conservation"].raw_features["pairwise_frequency_log_odds"] is None
    assert by_channel["conservation"].raw_features["pairwise_residual_log_odds"] is None
    assert by_channel["conservation"].score == by_channel["conservation"].raw_features[
        "independent_log_odds"
    ]
    assert "pseudocount_weight" in by_channel["conservation"].raw_features[
        "estimated_parameters"
    ]
    assert "pairwise_evolution_disabled_by_config" in by_channel[
        "conservation"
    ].warnings
    assert by_channel["structure"].quality_status == "ok"
    assert by_channel["structure"].raw_features["resource_id"] == "rcsb:1PGB"
    assert by_channel["structure"].raw_features["sites"]["39"]["structure_chain"] == "A"
    assert all(not by_channel[name].contributes_to_selection for name in by_channel if name != "kg")
    engine.close()
