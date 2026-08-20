from fitness_agents.agents.llm import MockScientistLLMClient
from fitness_agents.agents.scientist import ScientistAgent
from fitness_agents.contracts.schemas import CampaignState
from fitness_agents.data import load_dataset_bundle
from fitness_agents.protein_features import ProteinTaskContext


def _agent(experiment_config, *, knowledge_graph=None):
    return ScientistAgent(
        MockScientistLLMClient(),
        task_context=ProteinTaskContext.from_task(experiment_config.task),
        objective=experiment_config.task.objective,
        knowledge_graph=knowledge_graph,
    )


class _StubKnowledgeGraphTool:
    tool_name = "knowledge_graph"

    def __init__(self):
        self.calls = 0

    def hypothesis_context(self, *, round_id, limit=None):
        self.calls += 1
        return {
            "tool": self.tool_name,
            "query_id": f"kgq:r{round_id}",
            "beneficial_site_residues": [{"position": 39, "residue": "W", "support": 3}],
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
    agent = _agent(experiment_config)
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
    agent = _agent(experiment_config, knowledge_graph=tool)
    hypothesis = agent.propose_hypothesis(
        state, bundle.initial_variants, bundle.initial_observations, []
    )

    assert tool.calls == 1
    assert agent.last_knowledge_query_id == "kgq:r1"
    assert hypothesis.preferred_residues[39][0] == "W"
    assert "knowledge-graph query" in hypothesis.statement
    assert agent.inspect_variant(bundle.initial_variants[0].variant_id, round_id=1)["found"]


def test_scientist_repropose_after_critic_uses_attempt_id_and_parent(experiment_config):
    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path, experiment_config.task.oracle_data_path
    )
    state = CampaignState(run_id="test", mode="llm_agent", seed=1, round_id=1)
    agent = _agent(experiment_config)
    first = agent.propose_hypothesis(
        state, bundle.initial_variants, bundle.initial_observations, []
    )
    revised = agent.propose_hypothesis(
        state,
        bundle.initial_variants,
        bundle.initial_observations,
        [],
        critic_revision={
            "verdict": "REVISE",
            "summary": "Increase diversity and change the residue map.",
            "required_changes": [
                {"action": "REGENERATE_WITH_CONSTRAINTS", "rationale": "Restated prior batch."}
            ],
            "rejected_hypothesis_id": first.hypothesis_id,
            "rejected_statement": first.statement,
            "rejected_preferred_residues": {
                str(site): list(residues) for site, residues in first.preferred_residues.items()
            },
        },
        hypothesis_attempt=1,
    )
    assert first.hypothesis_id == "H01-00"
    assert revised.hypothesis_id == "H01-01"
    assert revised.parent_hypothesis_id == first.hypothesis_id
    assert "Revised after critic" in revised.statement
