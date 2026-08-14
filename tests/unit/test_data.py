from fitness_agents.data.gb1 import canonical_mutation_notation, variant_id
from fitness_agents.data.loader import load_dataset_bundle


def test_canonical_mutation_notation_is_reversible_and_stable():
    assert canonical_mutation_notation("VDGV") == "WT"
    assert canonical_mutation_notation("ADGA") == "V39A;V54A"
    assert variant_id("ADGA") == variant_id("ADGA")


def test_loader_keeps_hidden_pool_labels_out_of_public_bundle(synthetic_benchmark):
    bundle = load_dataset_bundle(synthetic_benchmark["public"], synthetic_benchmark["oracle"])
    assert len(bundle.initial_variants) == 24
    assert len(bundle.oracle_pool) == 88
    assert not hasattr(bundle.oracle_pool[0], "fitness")

