from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol

from stateguard.core.events import ReActStep
from stateguard.core.models import AgentAction, Message, ToolResult



@dataclass(frozen=True)
class NativeCodeActStep:
    action_name: str
    thought: str
    arguments: dict[str, Any]
    observation: str
    execution_attempted: bool
    execution_succeeded: bool
    terminal: bool = False
    final_output: str = ""
    raw_action: str = ""
    raw_observation: str = ""


class DECodeActSession(Protocol):
    def start(self, instruction: str) -> None: ...
    def advance(self) -> NativeCodeActStep: ...
    def inject_user_message(self, content: str) -> None: ...
    def snapshot(self) -> Any: ...
    def restore(self, snapshot: Any) -> None: ...
    def messages(self) -> tuple[Message, ...]: ...
    def trajectory(self) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class DEWorkerSnapshot:
    session: Any
    accepted_steps: int
    done: bool
    final_answer: str | None


class DACompDEWorkerAgent:
    """StateGuard Agent facade over DAComp's official CodeAct action/runtime pair."""

    def __init__(self, session: DECodeActSession, *, max_steps: int = 30) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.session = session
        self.max_steps = max_steps
        self._accepted_steps = 0
        self._done = False
        self._final_answer: str | None = None
        self._started = False

    @property
    def messages(self) -> tuple[Message, ...]:
        return self.session.messages() if self._started else ()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def final_answer(self) -> str | None:
        return self._final_answer

    @property
    def accepted_steps(self) -> int:
        return self._accepted_steps

    def start(self, prompt: Any, system_prompt: str | None = None) -> None:
        if self._started:
            raise RuntimeError("DE CodeAct Worker has already started")
        if system_prompt is not None:
            raise ValueError("DE uses CodeActAgent's official native system prompt")
        self.session.start(str(prompt))
        self._started = True

    def continue_turn(self, prompt: Any) -> None:
        del prompt
        raise RuntimeError("DAComp-DE is single-query")

    def step(self) -> ReActStep:
        if not self._started:
            raise RuntimeError("DE CodeAct Worker has not started")
        if self._done:
            raise RuntimeError("DE CodeAct Worker is already complete")
        if self._accepted_steps >= self.max_steps:
            self._done = True
            raise RuntimeError(
                f"DE CodeAct Worker exhausted the official {self.max_steps}-action budget"
            )
        native = self.session.advance()
        self._accepted_steps += 1
        budget_exhausted = self._accepted_steps >= self.max_steps
        self._done = native.terminal or budget_exhausted
        if native.terminal:
            self._final_answer = native.final_output
            action = AgentAction(
                kind="final", reasoning=native.thought, answer=native.final_output
            )
            observation = None
        else:
            action = AgentAction(
                kind="tool",
                reasoning=native.thought,
                tool_name=native.action_name,
                arguments=native.arguments,
            )
            observation = ToolResult(
                tool_name=native.action_name,
                ok=native.execution_succeeded,
                output=native.observation,
                data={
                    "execution_attempted": native.execution_attempted,
                    "execution_succeeded": native.execution_succeeded,
                    "native_action": native.raw_action,
                },
                error=None if native.execution_succeeded else native.observation,
            )
            if budget_exhausted:
                self._final_answer = ""
        return ReActStep(
            step_id=self._accepted_steps,
            action=action,
            observation=observation,
            done=self._done,
            raw_model_output=native.raw_action or native.thought,
            metadata={
                "benchmark": "dacomp",
                "track": "de",
                "official_action": native.raw_action,
                "official_observation": native.raw_observation or native.observation,
                "worker_budget_used": self._accepted_steps,
                "worker_budget_limit": self.max_steps,
            },
        )

    def inject_observation(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        del metadata
        self.session.inject_user_message(str(content))
        if self._done and self._accepted_steps < self.max_steps:
            self._done = False
            self._final_answer = None

    def snapshot(self) -> DEWorkerSnapshot:
        return DEWorkerSnapshot(
            self.session.snapshot(),
            self._accepted_steps,
            self._done,
            self._final_answer,
        )

    def restore(self, snapshot: DEWorkerSnapshot) -> None:
        self.session.restore(snapshot.session)
        self._accepted_steps = snapshot.accepted_steps
        self._done = snapshot.done
        self._final_answer = snapshot.final_answer

    def trajectory(self) -> list[dict[str, Any]]:
        return self.session.trajectory()

    def close(self) -> None:
        self.session.close()
