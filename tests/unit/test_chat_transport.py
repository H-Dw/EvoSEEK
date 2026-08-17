from __future__ import annotations

from fitness_agents.agents.transports import MockChatTransport


def test_mock_chat_transport_is_deterministic_and_records_requests() -> None:
    transport = MockChatTransport([{"response": 1}])
    assert transport.create_chat_completion(model="model") == {"response": 1}
    assert transport.calls == [{"model": "model"}]
