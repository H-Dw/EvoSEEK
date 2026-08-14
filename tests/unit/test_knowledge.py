from fitness_agents.data import load_dataset_bundle
from fitness_agents.knowledge import KnowledgeEngine


def test_knowledge_channels_are_independently_switchable(experiment_config, tmp_path):
    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path, experiment_config.task.oracle_data_path
    )
    engine = KnowledgeEngine(
        experiment_config.knowledge, graph_path=tmp_path / "kg.sqlite", assay_id="test"
    )
    engine.update(bundle.initial_variants, bundle.initial_observations)
    evidence = engine.evidence_for(bundle.oracle_pool[:2], round_id=1)
    assert {item.channel for item in evidence[bundle.oracle_pool[0].variant_id]} == {
        "physchem", "conservation", "structure", "kg"
    }
    assert engine.evidence_for(bundle.oracle_pool[:2], round_id=1, delete_evidence=True) == {}
    assert all(edge["predicate"] == "OBSERVED_IN_CONTEXT" for edge in engine.graph.export_edges())
    engine.close()

