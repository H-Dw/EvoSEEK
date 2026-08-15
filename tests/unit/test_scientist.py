from fitness_agents.agents.llm import MockScientistLLMClient
from fitness_agents.agents.scientist import ScientistAgent
from fitness_agents.contracts.schemas import CampaignState
from fitness_agents.data import load_dataset_bundle


class _StubKnowledgeGraphTool:
    tool_name = "knowledge_graph"

    def __init__(self):
        self.calls = 0

    def hypothesis_context(self, *, round_id, limit=None):
        self.calls += 1
        return {
            "tool": self.tool_name,
            "query_id": f"kgq:r{round_id}",
            "beneficial_site_residues": [
                {"position": 39, "residue": "W", "support": 3}
            ],
        }

    def explain_variant(self, variant_id, *, round_id):
        return {
            "tool": self.tool_name,
            "query_id": f"kgq:{variant_id}:r{round_id}",
            "variant_id": variant_id,
            "found": True,
        }


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


def test_scientist_agent_calls_configured_knowledge_graph(experiment_config):
    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path, experiment_config.task.oracle_data_path
    )
    state = CampaignState(run_id="test", mode="knowledge_agent", seed=1, round_id=1)
    tool = _StubKnowledgeGraphTool()
    agent = ScientistAgent(MockScientistLLMClient(), knowledge_graph=tool)
    hypothesis = agent.propose_hypothesis(
        state, bundle.initial_variants, bundle.initial_observations, []
    )

    assert tool.calls == 1
    assert agent.last_knowledge_query_id == "kgq:r1"
    assert hypothesis.preferred_residues[39][0] == "W"
    assert "knowledge-graph query" in hypothesis.statement
    assert agent.inspect_variant(bundle.initial_variants[0].variant_id, round_id=1)["found"]
