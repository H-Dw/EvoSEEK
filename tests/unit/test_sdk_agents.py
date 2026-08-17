from dataclasses import replace

from fitness_agents.agents.llm import MockScientistLLMClient, create_llm_client
from fitness_agents.agents.output_contracts import HypothesisOutput
from fitness_agents.agents.rethink import MockReThinkClient, create_rethink_client
from fitness_agents.agents.scientist import ScientistAgent
from fitness_agents.agents.sdk_agents import AgentsSDKReThinkClient, AgentsSDKScientistLLMClient
from fitness_agents.config import load_experiment_config
from fitness_agents.contracts.schemas import CampaignState, Evidence
from fitness_agents.data import load_dataset_bundle


def _hypothesis_payload(hypothesis_id: str, parent_id: str | None = None) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "statement": "Visible observations support testing W at site 39.",
        "preferred_residues": {"39": ["W"], "40": ["D"], "41": ["G"], "54": ["V"]},
        "evidence_ids": [],
        "expected_outcome": "The batch should enrich high-fitness variants relative to random selection.",
        "falsification_criterion": "Reject or revise if the revealed batch median fails to exceed the visible median.",
        "parent_hypothesis_id": parent_id,
    }


def test_mock_provider_ignores_sdk_runtime():
    assert isinstance(create_llm_client("mock", runtime="agents_sdk"), MockScientistLLMClient)
    assert isinstance(create_rethink_client("mock", runtime="agents_sdk"), MockReThinkClient)


def test_factory_selects_sdk_clients_without_importing_openai_agents():
    scientist = create_llm_client(
        "deepseek",
        runtime="agents_sdk",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key="env:DEEPSEEK_API_KEY",
    )
    rethink = create_rethink_client(
        "deepseek",
        runtime="agents_sdk",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key="env:DEEPSEEK_API_KEY",
    )
    assert isinstance(scientist, AgentsSDKScientistLLMClient)
    assert scientist.supports_kg_tools is True
    assert isinstance(rethink, AgentsSDKReThinkClient)


def test_al96_sdk_config_keeps_manifest_fold_and_agents_runtime():
    config = load_experiment_config("configs/experiments/knowledge_agent_al96_sdk.yaml")
    assert config.llm.runtime == "agents_sdk"
    assert config.llm.provider == "deepseek"
    assert config.llm.api_key == "env:DEEPSEEK_API_KEY"
    assert config.task.split_root is not None
    assert config.generation.selection_driver == "agent_uq"
    random_config = load_experiment_config("configs/experiments/random_al96.yaml")
    fitness_config = load_experiment_config("configs/experiments/fitness_direct_al96.yaml")
    assert random_config.mode == "random"
    assert random_config.llm.provider == "mock"
    assert fitness_config.mode == "fitness_direct"
    assert fitness_config.generation.selection_driver == "predictor"
    assert random_config.budget_per_round == fitness_config.budget_per_round == config.budget_per_round
    assert random_config.task.split_root == config.task.split_root


def test_llm_runtime_override_switches_existing_al96_config_to_sdk():
    config = load_experiment_config(
        "configs/experiments/knowledge_agent_al96.yaml",
        overrides={"llm": {"runtime": "agents_sdk"}},
    )
    assert config.llm.runtime == "agents_sdk"
    assert config.llm.provider == "deepseek"
    assert config.llm.model == "deepseek-v4-flash"


def test_sdk_scientist_uses_injected_runner_and_records_tool_session_query_ids(
    experiment_config, monkeypatch
):
    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path, experiment_config.task.oracle_data_path
    )
    state = CampaignState(run_id="sdk-test", mode="knowledge_agent", seed=1, round_id=1)
    client = AgentsSDKScientistLLMClient(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key="env:DEEPSEEK_API_KEY",
        provider="deepseek",
    )

    def fake_run_sdk(self, **kwargs):
        assert kwargs["role"] == "scientist"
        assert kwargs["kg_tool_session"] is session
        expected_id = f"hyp:{state.run_id}:r{state.round_id}"
        return _hypothesis_payload(expected_id)

    monkeypatch.setattr(AgentsSDKScientistLLMClient, "_run_sdk", fake_run_sdk)

    class _Session:
        query_ids = ("kgq:sdk:1",)

    session = _Session()
    agent = ScientistAgent(client)
    hypothesis = agent.propose_hypothesis(
        state,
        bundle.initial_variants,
        bundle.initial_observations,
        (),
        kg_tool_session=session,
    )
    parsed = HypothesisOutput.model_validate(
        _hypothesis_payload(hypothesis.hypothesis_id)
    ).to_hypothesis(expected_hypothesis_id=hypothesis.hypothesis_id)
    assert hypothesis.hypothesis_id == parsed.hypothesis_id
    assert hypothesis.preferred_residues[39] == ("W",)
    assert agent.last_knowledge_query_ids == ("kgq:sdk:1",)


def test_sdk_scientist_rejects_evidence_ids_outside_visible_set(monkeypatch):
    client = AgentsSDKScientistLLMClient(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key="env:DEEPSEEK_API_KEY",
        provider="deepseek",
    )
    evidence = (
        Evidence("ev:visible", "var:1", "physchem", "visible", 0.2, "test", 0.8, 1),
    )

    def fake_run_sdk(self, **kwargs):
        payload = _hypothesis_payload("hyp:run:r1")
        payload["evidence_ids"] = ["ev:hidden"]
        return kwargs["validator"](payload)

    monkeypatch.setattr(AgentsSDKScientistLLMClient, "_run_sdk", fake_run_sdk)
    try:
        client.generate_hypothesis(
            sanitized_context={
                "expected_hypothesis_id": "hyp:run:r1",
                "previous_hypothesis_id": None,
            },
            evidence=evidence,
            output_schema={},
        )
    except ValueError as error:
        assert "not visible" in str(error)
        return
    raise AssertionError("expected evidence-id contract to fail")


def test_sdk_baseline_loader_pairs_random_and_fitness_direct_on_the_same_fold():
    random_config = load_experiment_config(
        "configs/experiments/random_al96.yaml",
        overrides={"seed": 42, "run_label": "al96-sdk-baseline"},
    )
    fitness_config = load_experiment_config(
        "configs/experiments/fitness_direct_al96.yaml",
        overrides={"seed": 42, "run_label": "al96-sdk-baseline"},
    )
    agent_config = load_experiment_config(
        "configs/experiments/knowledge_agent_al96_sdk.yaml",
        overrides={"seed": 42, "run_label": "al96-sdk-baseline", "llm": {"runtime": "agents_sdk"}},
    )
    random_config = replace(random_config, task=replace(random_config.task, fold_index=1))
    fitness_config = replace(fitness_config, task=replace(fitness_config.task, fold_index=1))
    agent_config = replace(agent_config, task=replace(agent_config.task, fold_index=1))
    assert random_config.mode == "random"
    assert fitness_config.mode == "fitness_direct"
    assert agent_config.llm.runtime == "agents_sdk"
    assert agent_config.task.fold_index == fitness_config.task.fold_index == 1
    assert random_config.seed == fitness_config.seed == agent_config.seed == 42
    assert (
        random_config.rounds * random_config.budget_per_round
        == fitness_config.rounds * fitness_config.budget_per_round
        == agent_config.rounds * agent_config.budget_per_round
    )
