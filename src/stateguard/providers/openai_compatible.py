from __future__ import annotations

import json
import urllib.request

from stateguard.core.models import Message

from .base import ModelResponse


class OpenAICompatibleClient:
    """Dependency-free adapter for chat-completions compatible endpoints."""

    def __init__(self, model: str, api_base: str, api_key: str, timeout: float = 300.0) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        payload: dict = {
            "model": self.model,
            "messages": [
                {key: value for key, value in {"role": m.role, "content": m.content, "name": m.name}.items() if value is not None}
                for m in messages
            ],
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": schema} for schema in tools
            ]
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        message = body["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or content
        # A native tool_calls entry always wins over content. Reasoning models
        # return prose alongside the call, and parsing that prose as the action
        # yields an invalid envelope. Prose and reasoning_content stay as the
        # reasoning. Mirrors the DSBench adapter client.
        if message.get("tool_calls"):
            call = message["tool_calls"][0]["function"]
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            content = json.dumps(
                {
                    "type": "tool",
                    "reasoning": reasoning,
                    "tool": call["name"],
                    "arguments": arguments,
                },
                ensure_ascii=False,
            )
        # Carry finish_reason alongside usage so a truncated action can be told
        # apart from one the model chose to end: "length" means the completion
        # budget ran out, "stop" means it stopped on its own.
        metadata = dict(body.get("usage") or {})
        metadata["finish_reason"] = body["choices"][0].get("finish_reason")
        return ModelResponse(content=content, reasoning=reasoning, metadata=metadata)
