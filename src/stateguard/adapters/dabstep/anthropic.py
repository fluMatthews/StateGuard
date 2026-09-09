from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

try:
    from smolagents.models import ChatMessage, Model
except ImportError as exc:  # pragma: no cover - guarded by the DABstep extra
    raise ImportError("DABstep Anthropic support requires the 'dabstep' extra") from exc


_FINALIZER_SYSTEM_PREFIX = "An agent tried to answer a user query but it got stuck"
_NORMAL_FORMAT_CORRECTION = (
    "Format correction only: the environment accepts textual Python Code blocks, "
    "not structured tool calls or Bash. Re-express the same intended action using "
    "the required `Code: ```py ... ```` text format. Do not change the analysis."
)
_FINAL_FORMAT_CORRECTION = (
    "Format correction only: return the best final answer available from the existing "
    "memory as plain text satisfying the original answer guidelines. Do not call a tool, "
    "write code, or describe another analysis step, even if the computation is incomplete."
)


class _ResponseFormatError(RuntimeError):
    pass


class AnthropicMessagesModel(Model):
    """Anthropic Messages transport preserving DABstep's text CodeAgent contract."""

    def __init__(
        self,
        *,
        model_id: str,
        api_base: str,
        api_key: str,
        max_tokens: int = 3000,
        timeout: float = 300.0,
        max_attempts: int = 3,
    ) -> None:
        super().__init__()
        self.model_id = model_id.removeprefix("anthropic/")
        self.endpoint = _messages_endpoint(api_base)
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.last_input_token_count = 0
        self.last_output_token_count = 0

    def __call__(
        self,
        messages: list[dict[str, Any]],
        stop_sequences: list[str] | None = None,
        grammar: str | None = None,
        max_tokens: int | None = None,
        **_: Any,
    ) -> ChatMessage:
        del grammar
        system, native_messages = _convert_messages(messages)
        finalizer = system.startswith(_FINALIZER_SYSTEM_PREFIX)
        token_limit = max_tokens or self.max_tokens
        data = self._complete(system, native_messages, stop_sequences, token_limit)
        try:
            text = _render_response(data, finalizer=finalizer)
        except _ResponseFormatError:
            correction = _FINAL_FORMAT_CORRECTION if finalizer else _NORMAL_FORMAT_CORRECTION
            corrected_messages = _append_user(native_messages, correction)
            data = self._complete(
                system,
                corrected_messages,
                stop_sequences,
                min(token_limit, 800) if finalizer else token_limit,
            )
            text = _render_response(data, finalizer=finalizer)
        return ChatMessage(role="assistant", content=text)

    def _complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        stop_sequences: list[str] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            # This intermediary injects hosted Python/Bash tools when this key
            # is omitted. Keep it explicit for the text-only DABstep worker.
            "tools": [],
        }
        if system:
            payload["system"] = system
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences
        data = self._request(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        usage = data.get("usage") or {}
        self.last_input_token_count = int(usage.get("input_tokens") or 0)
        self.last_output_token_count = int(usage.get("output_tokens") or 0)
        return data

    def _request(self, body: bytes) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            request = urllib.request.Request(
                self.endpoint,
                data=body,
                headers={
                    "content-type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    detail = exc.read().decode("utf-8", errors="replace")[:1000]
                    raise RuntimeError(f"Anthropic HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
            if attempt + 1 < self.max_attempts:
                time.sleep(2**attempt)
        raise RuntimeError(f"Anthropic request failed after {self.max_attempts} attempts: {last_error}")


def _render_response(data: dict[str, Any], *, finalizer: bool) -> str:
    rendered: list[str] = []
    for block in data.get("content", []):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            rendered.append(str(block.get("text", "")))
            continue
        if block_type != "tool_use":
            continue
        if finalizer:
            raise _ResponseFormatError("finalizer returned a tool call")
        tool_name = str(block.get("name", ""))
        if tool_name != "python_interpreter":
            raise _ResponseFormatError(f"unsupported hosted tool: {tool_name}")
        code = _tool_code(block.get("input"))
        if not code:
            raise _ResponseFormatError("python_interpreter tool call has no code")
        rendered.append(f"\n\nCode:\n```py\n{code}\n```\n<end_code>")
    text = "".join(rendered).strip()
    if not text:
        raise _ResponseFormatError("response contained no usable text or Python action")
    if finalizer and _looks_like_action(text):
        raise _ResponseFormatError("finalizer continued the ReAct trajectory")
    return text


def _looks_like_action(text: str) -> bool:
    return bool(
        "```" in text
        or re.search(r"(^|\n)\s*(?:Thought|Code)\s*:", text, flags=re.IGNORECASE)
        or "python_interpreter" in text
    )


def _tool_code(tool_input: Any) -> str:
    if isinstance(tool_input, str):
        return tool_input.strip()
    if not isinstance(tool_input, dict):
        return ""
    value = tool_input.get("code") or tool_input.get("arguments")
    if value is None and len(tool_input) == 1:
        value = next(iter(tool_input.values()))
    return value.strip() if isinstance(value, str) else ""


def _append_user(messages: list[dict[str, str]], content: str) -> list[dict[str, str]]:
    result = [dict(message) for message in messages]
    if result and result[-1]["role"] == "user":
        result[-1]["content"] += "\n\n" + content
    else:
        result.append({"role": "user", "content": content})
    return result


def _messages_endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def _convert_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    converted: list[dict[str, str]] = []
    for message in messages:
        raw_role = message.get("role", "user")
        role = str(getattr(raw_role, "value", raw_role))
        content = _content_text(message.get("content"))
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        native_role = "assistant" if role == "assistant" else "user"
        if converted and converted[-1]["role"] == native_role:
            converted[-1]["content"] += "\n\n" + content
        else:
            converted.append({"role": native_role, "content": content})
    if not converted:
        converted.append({"role": "user", "content": "Continue."})
    return "\n\n".join(system_parts), converted


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(json.dumps(block, ensure_ascii=False, default=str))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content)
