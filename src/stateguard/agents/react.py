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
Return exactly one JSON object per step.
Tool action: {"type":"tool","reasoning":"...","tool":"name","arguments":{...},"metadata":{}}
Final action: {"type":"final","reasoning":"...","answer":"...","metadata":{}}
Use tools for computations and never claim an execution that did not occur.
When a python tool is available, use it for code execution and inspect its real observation.
"""


class ActionParseError(ValueError):
    pass


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


def parse_action(text: str) -> AgentAction:
    value = parse_json_object(text)
    kind = str(value.get("type", value.get("kind", ""))).lower()
    if kind == "tool":
        arguments = value.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ActionParseError("tool arguments must be an object")
        return AgentAction(
            kind="tool",
            reasoning=str(value.get("reasoning", "")),
            tool_name=str(value.get("tool", value.get("tool_name", ""))),
            arguments=arguments,
            metadata=dict(value.get("metadata", {})),
        )
    if kind == "final":
        answer = value.get("answer", "")
        if not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False)
        return AgentAction(
            kind="final",
            reasoning=str(value.get("reasoning", "")),
            answer=answer,
            metadata=dict(value.get("metadata", {})),
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

    def step(self) -> ReActStep:
        if not self.session.messages:
            raise RuntimeError("agent has not been started")
        if self.session.done:
            raise RuntimeError("agent session is already complete")
        if self.session.step_count >= self.max_steps:
            raise RuntimeError(f"agent exceeded max_steps={self.max_steps}")

        response = self.model.complete(list(self.session.messages), self.tools.schemas())
        action = parse_action(response.content)
        self.session.step_count += 1
        self.session.messages.append(
            Message("assistant", response.content, metadata={"reasoning": response.reasoning})
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
        }
