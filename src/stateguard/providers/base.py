from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from stateguard.core.models import Message


@dataclass(frozen=True)
class ModelResponse:
    content: str
    reasoning: str = ""
    metadata: dict | None = None


class ModelClient(Protocol):
    def complete(self, messages: list[Message], tools: list[dict]) -> ModelResponse: ...


class ScriptedModelClient:
    """Deterministic provider used by examples and tests."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.index = 0

    def complete(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        del messages, tools
        if self.index >= len(self.responses):
            raise RuntimeError("scripted model exhausted")
        response = self.responses[self.index]
        self.index += 1
        return ModelResponse(response)
