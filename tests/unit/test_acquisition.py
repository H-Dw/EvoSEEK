import numpy as np

from fitness_agents.acquisition import GreedyPolicy, UCBPolicy
from fitness_agents.contracts.schemas import CampaignState, Prediction, Variant
from fitness_agents.mutation.generators import EnumeratingCandidateGenerator


def _prediction(identifier: str, mean: float, std: float) -> Prediction:
    return Prediction(identifier, mean, std, (mean - std, mean + std), 0.0, {}, "test")


def _variant(identifier: str, code: str) -> Variant:
    return Variant(identifier, code, code, code, 4, "oracle_pool")


def test_ucb_explores_uncertain_candidate_while_greedy_exploits():
    predictions = [_prediction("a", 1.0, 0.01), _prediction("b", 0.8, 0.4)]
    rng = np.random.default_rng(1)
    greedy = GreedyPolicy().score(predictions, {}, rng)
    ucb = UCBPolicy(beta=2.0).score(predictions, {}, rng)
    assert max(greedy, key=greedy.get) == "a"
    assert max(ucb, key=ucb.get) == "b"


def test_batch_diversity_can_avoid_near_duplicate():
    variants = [_variant("a", "AAAA"), _variant("b", "AAAV"), _variant("c", "VVVV")]
    predictions = [_prediction("a", 1.0, 0.1), _prediction("b", 0.99, 0.1), _prediction("c", 0.95, 0.1)]
    policy = GreedyPolicy()
    scores = policy.score(predictions, {}, np.random.default_rng(1))
    assert policy.select(variants, predictions, scores, 2, diversity_lambda=0.2) == ["a", "c"]


def test_enumerating_generator_respects_positive_candidate_limit():
    variants = [_variant("a", "AAAA"), _variant("b", "AAAV"), _variant("c", "VVVV")]
    generator = EnumeratingCandidateGenerator()
    state = CampaignState(run_id="t", mode="fitness_direct", seed=1)
    assert generator.generate(variants, state, None, {}, 0) == variants
    assert generator.generate(variants, state, None, {}, 2) == variants[:2]

