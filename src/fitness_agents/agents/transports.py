"""Minimal provider transport boundary owned by this project."""

from __future__ import annotations

from typing import Any, Protocol


class ChatTransport(Protocol):
    base_url: str

    def create_chat_completion(self, **kwargs: Any) -> Any: ...


class OpenAICompatibleChatTransport:
    """Use the base OpenAI package only as an HTTP Chat Completions client."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.base_url = str(getattr(client, "base_url", "") or "")

    def create_chat_completion(self, **kwargs: Any) -> Any:
        return self.client.chat.completions.create(**kwargs)


class MockChatTransport:
    def __init__(self, responses: list[Any], *, base_url: str = "mock://local") -> None:
        self.responses = list(responses)
        self.base_url = base_url
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("Mock transport response queue is empty")
        return self.responses.pop(0)
