from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict
from typing import Any

from stateguard.core.events import ReActStep
from stateguard.core.models import AgentAction, AgentSession, Message
from stateguard.providers.base import ModelClient
from stateguard.runtime.tools import ToolRegistry


DEFAULT_REACT_SYSTEM_PROMPT = """You are a tool-using ReAct agent.

Follow the task's explicit requirements and use the available tools whenever
external data or computation is needed.

Return exactly one JSON action per step.

Tool action:
{"type":"tool","reasoning":"why this tool call is needed","tool":"tool_name","arguments":{...}}

Final action:
{"type":"final","reasoning":"why the answer is supported","answer":"final answer"}

Emit only one action at a time. After a tool action, wait for and inspect the
actual tool result before deciding the next action. Never claim that code ran
unless a successful tool result was returned. Do not invent tool outputs.
"""


class ActionParseError(ValueError):
    """Raised when a model response cannot be read as an action.

    The response never reaches ``session.messages`` -- parsing happens before the
    append, and a malformed action must not become part of the agent's context.
    Carrying the text on the exception is therefore the only way a caller can
    still record what the model actually emitted.
    """

    def __init__(self, message: str, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ActionParseError("model response contains no JSON object")
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"invalid action JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ActionParseError("action must be a JSON object")
    return value


def _is_native_tool_call(value: dict[str, Any]) -> bool:
    """True for a provider tool-call envelope that only lacks its action type."""
    return (
        isinstance(value.get("name"), str)
        and bool(value["name"])
        and isinstance(value.get("arguments"), dict)
    )


def parse_action(text: str) -> AgentAction:
    value = parse_json_object(text)
    kind = str(value.get("type", value.get("kind", ""))).lower()
    if not kind and _is_native_tool_call(value):
        # A reasoning model served with tool schemas sometimes answers with the
        # provider's own call envelope -- {"name": ..., "arguments": {...}} --
        # written into the message content rather than into tool_calls, where
        # OpenAICompatibleClient would have converted it. The call is complete
        # and unambiguous; only the "type" the text protocol asks for is
        # missing, so reading it as a tool action loses nothing.
        return AgentAction(
            kind="tool",
            reasoning=str(value.get("reasoning", "")),
            tool_name=str(value["name"]),
            arguments=value["arguments"],
        )
    if kind == "tool":
        arguments = value.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ActionParseError("tool arguments must be an object")
        return AgentAction(
            kind="tool",
            reasoning=str(value.get("reasoning", "")),
            tool_name=str(value.get("tool", value.get("tool_name", ""))),
            arguments=arguments,
        )
    if kind == "final":
        answer = value.get("answer", "")
        if not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False)
        return AgentAction(
            kind="final",
            reasoning=str(value.get("reasoning", "")),
            answer=answer,
        )
    if kind == "control":
        answer = value.get("answer", "")
        if not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False)
        return AgentAction(
            kind="control",
            reasoning=str(value.get("reasoning", "")),
            answer=answer,
        )
    raise ActionParseError(f"unsupported action type: {kind!r}")


class ReActAgent:
    """Role-neutral, interruptible ReAct engine."""

    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry | None = None,
        *,
        system_prompt: str = DEFAULT_REACT_SYSTEM_PROMPT,
        max_steps: int = 40,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.model = model
        self.tools = tools.clone() if tools is not None else ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.session = AgentSession()
        # The server reports what a request actually cost and why generation
        # stopped. Nothing kept it, so a truncated reply reached the harness as
        # an unparseable string and its size had to be inferred from characters.
        self.last_model_metadata: dict[str, Any] = {}

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self.session.messages)

    @property
    def done(self) -> bool:
        return self.session.done

    @property
    def final_answer(self) -> str | None:
        return self.session.final_answer

    def start(self, prompt: str, system_prompt: str | None = None) -> None:
        self.session = AgentSession(
            messages=[
                Message("system", system_prompt or self.system_prompt),
                Message("user", prompt),
            ]
        )

    def continue_turn(self, prompt: str) -> None:
        """Append a new user turn while preserving the complete prior trajectory."""
        if not self.session.messages:
            raise RuntimeError("agent has not been started")
        if not self.session.done:
            raise RuntimeError("cannot begin a new turn before the current turn is complete")
        self.session.messages.append(Message("user", prompt, metadata={"new_turn": True}))
        # DSGym resets only the per-turn step budget; conversation and tool
        # workspace remain intact across turns.
        self.session.step_count = 0
        self.session.done = False
        self.session.final_answer = None

    def inject_observation(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        self.session.messages.append(Message("user", content, metadata=metadata or {}))
        self.session.done = False
        self.session.final_answer = None

    def _messages_for_model(self) -> list[Message]:
        """Return the model payload; subclasses may apply role-specific context policy."""
        return list(self.session.messages)

    def step(self) -> ReActStep:
        if not self.session.messages:
            raise RuntimeError("agent has not been started")
        if self.session.done:
            raise RuntimeError("agent session is already complete")
        if self.session.step_count >= self.max_steps:
            raise RuntimeError(f"agent exceeded max_steps={self.max_steps}")

        response = self.model.complete(self._messages_for_model(), self.tools.schemas())
        self.last_model_metadata = dict(response.metadata or {})
        try:
            action = parse_action(response.content)
        except ActionParseError as exc:
            # Attach the unparsed text so the harness can record it. The session
            # itself stays clean: a malformed action must not enter the context.
            if exc.raw_response is None:
                exc.raw_response = response.content
            raise
        self.session.step_count += 1
        self.session.messages.append(
            Message(
                "assistant",
                response.content,
                metadata={"reasoning": response.reasoning, **self.last_model_metadata},
            )
        )

        observation = None
        if action.kind == "tool":
            observation = self.tools.execute(action.tool_name or "", action.arguments)
            rendered = {
                "tool": observation.tool_name,
                "ok": observation.ok,
                "output": observation.output,
                "data": observation.data,
                "error": observation.error,
            }
            self.session.messages.append(
                Message(
                    "user",
                    "<tool_result>\n"
                    + json.dumps(rendered, ensure_ascii=False, default=str)
                    + "\n</tool_result>",
                    name=observation.tool_name,
                )
            )
        else:
            self.session.done = True
            self.session.final_answer = action.answer

        return ReActStep(
            step_id=self.session.step_count,
            action=action,
            observation=observation,
            done=self.session.done,
            raw_model_output=response.content,
        )

    def run(self) -> str:
        while not self.done:
            self.step()
        return self.final_answer or ""

    def run_for(self, max_new_steps: int) -> str:
        """Run one bounded activation while preserving the long-lived session."""
        if max_new_steps < 1:
            raise ValueError("max_new_steps must be positive")
        started = self.session.step_count
        while not self.done and self.session.step_count - started < max_new_steps:
            self.step()
        if not self.done:
            raise RuntimeError(f"agent activation exceeded {max_new_steps} ReAct steps")
        return self.final_answer or ""

    def snapshot(self) -> AgentSession:
        return self.session.snapshot()

    def restore(self, snapshot: AgentSession) -> None:
        self.session = copy.deepcopy(snapshot)

    def export_session(self) -> dict[str, Any]:
        return {
            "messages": [asdict(message) for message in self.messages],
            "step_count": self.session.step_count,
            "done": self.done,
            "final_answer": self.final_answer,
            "last_model_metadata": self.last_model_metadata,
        }
