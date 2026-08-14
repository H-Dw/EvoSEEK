from fitness_agents.agents.llm import MockScientistLLMClient
from fitness_agents.agents.scientist import ScientistAgent
from fitness_agents.contracts.schemas import CampaignState
from fitness_agents.data import load_dataset_bundle


def test_mock_scientist_produces_falsifiable_updated_hypothesis(experiment_config):
    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path, experiment_config.task.oracle_data_path
    )
    state = CampaignState(run_id="test", mode="llm_agent", seed=1, round_id=1)
    agent = ScientistAgent(MockScientistLLMClient())
    hypothesis = agent.propose_hypothesis(
        state, bundle.initial_variants, bundle.initial_observations, []
    )
    assert hypothesis.preferred_residues
    assert hypothesis.expected_outcome
    assert "Reject or revise" in hypothesis.falsification_criterion

