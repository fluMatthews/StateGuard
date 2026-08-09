from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

from stateguard.core.events import ReActStep
from stateguard.core.models import AgentAction, Message, ToolResult

from .messages import (
    core_messages,
    extract_answer,
    extract_python,
    official_messages,
    reasoning_text,
    wrap_stateguard_observation,
)
from .workspace import LongDSWorkspace


@dataclass(frozen=True)
class LongDSWorkerSnapshot:
    conversation: tuple[dict[str, str], ...]
    turn_step_count: int
    done: bool
    final_answer: str | None
    raw_responses: tuple[str, ...]
    completed_step_count: int
    turn_token_count: int


class LongDSWorkerAgent:
    """Agent protocol wrapper around DSGym's native backend and AllocatedCodeEnv."""

    def __init__(
        self,
        *,
        backend: Any,
        environment: Any,
        workspace: LongDSWorkspace,
        clean_output: Any,
        max_steps_per_turn: int = 40,
    ) -> None:
        if max_steps_per_turn < 1:
            raise ValueError("max_steps_per_turn must be positive")
        self.backend = backend
        self.environment = environment
        self.workspace = workspace
        self.max_steps_per_turn = max_steps_per_turn
        self.conversation: list[dict[str, str]] = []
        self.turn_step_count = 0
        self._done = False
        self._final_answer: str | None = None
        self.raw_responses: list[str] = []
        self.completed_step_count = 0
        self.turn_token_count = 0
        self.workspace.bind_environment(environment, clean_output)

    @property
    def messages(self) -> tuple[Message, ...]:
        return core_messages(self.conversation)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def final_answer(self) -> str | None:
        return self._final_answer

    def start(self, prompt: Any, system_prompt: str | None = None) -> None:
        if self.conversation:
            raise RuntimeError("LongDS Worker has already started")
        if system_prompt is not None:
            raise ValueError("LongDS system prompt must come from the official native messages")
        messages = _coerce_messages(prompt)
        self.conversation, metadata = self.environment.init(messages)
        _require_execution_environment(self.environment, metadata)
        # Official env.init returns its input list. Keep one explicit source of truth.
        self.conversation = official_messages(self.conversation)
        self.turn_step_count = 0
        self._done = False
        self._final_answer = None
        self.completed_step_count = 0
        self.turn_token_count = 0

    def continue_turn(self, prompt: Any) -> None:
        if not self.conversation:
            raise RuntimeError("LongDS Worker has not started")
        if not self._done:
            raise RuntimeError("cannot begin a LongDS turn before the prior turn completes")
        self.environment.reset_turns()
        for message in _coerce_messages(prompt):
            if message["role"] == "user":
                self.conversation.append(message)
        self.environment.chat_history = copy.deepcopy(self.conversation)
        self.turn_step_count = 0
        self._done = False
        self._final_answer = None
        self.completed_step_count = 0
        self.turn_token_count = 0

    def reset_environment_preserve_conversation(self) -> None:
        """Match DSGym reset_env_times: new container, same LLM transcript."""
        if not self.conversation:
            raise RuntimeError("LongDS Worker has not started")
        self.environment.close()
        self.workspace.executions = []
        self.workspace.operations = []
        self.workspace.cleanup_events = []
        restored, metadata = self.environment.init(copy.deepcopy(self.conversation))
        _require_execution_environment(self.environment, metadata)
        self.conversation = official_messages(restored)
        self.environment.chat_history = copy.deepcopy(self.conversation)

    def step(self) -> ReActStep:
        if not self.conversation:
            raise RuntimeError("LongDS Worker has not started")
        if self._done:
            raise RuntimeError("LongDS Worker turn is already complete")
        if self.turn_step_count >= self.max_steps_per_turn:
            raise RuntimeError("LongDS Worker exhausted its per-turn budget")

        self.turn_step_count += 1
        try:
            raw_response = self.backend.generate(self.conversation)
            self.raw_responses.append(raw_response)
            self.turn_token_count += len(raw_response.split())
            execution_count = len(self.workspace.executions)
            output = self.environment.step(raw_response)
            self.completed_step_count += 1
            action_text = output.get("postprocessed_action", raw_response)
            if not str(action_text).strip():
                # Match official runner fallback without fabricating an answer/tool result.
                action_text = "<reasoning>no postprocessed action</reasoning>"
            self.conversation.append({"role": "assistant", "content": str(action_text)})
            observations = official_messages(output.get("observations", []))
            self.conversation.extend(observations)
            self.environment.chat_history = copy.deepcopy(self.conversation)

            answer = extract_answer(str(action_text))
            code = extract_python(str(action_text))
            env_done = bool(output.get("done"))
            budget_exhausted = self.turn_step_count >= self.max_steps_per_turn
            self._done = env_done or budget_exhausted
            metadata = dict(output.get("metadata") or {})
            official_final = metadata.get("final_answer")
            if answer is not None:
                self._final_answer = str(official_final or answer)
                completion_reason = "answer"
            elif self._done:
                self._final_answer = ""
                completion_reason = "budget_exhausted"
            else:
                completion_reason = "continue"

            observation: ToolResult | None = None
            if answer is not None:
                action = AgentAction(
                    kind="final",
                    reasoning=reasoning_text(str(action_text)),
                    answer=self._final_answer,
                )
            else:
                action = AgentAction(
                    kind="tool",
                    reasoning=reasoning_text(str(action_text)),
                    tool_name="python",
                    arguments={"code": code or ""},
                )
                if code is not None and len(self.workspace.executions) > execution_count:
                    execution = self.workspace.executions[-1]
                    observation = ToolResult(
                        tool_name="python",
                        ok=execution.succeeded,
                        output=execution.cleaned_output,
                        data={
                            "execution_attempted": execution.attempted,
                            "execution_succeeded": execution.succeeded,
                            "raw_outputs": list(execution.raw_outputs),
                        },
                        error=execution.error,
                    )
                elif code is not None:
                    # On DSGym's last allowed step env.step marks done before execution.
                    observation = None
                else:
                    rendered = observations[0]["content"] if observations else ""
                    observation = ToolResult(
                        tool_name="python",
                        ok=False,
                        output=rendered,
                        data={
                            "execution_attempted": False,
                            "execution_succeeded": False,
                        },
                        error="No python code found",
                    )

            return ReActStep(
                step_id=self.turn_step_count,
                action=action,
                observation=observation,
                done=self._done,
                # Manager sees the action DSGym actually accepted/executed, not any
                # trailing model text discarded by official post-processing.
                raw_model_output=str(action_text),
                metadata={
                    "benchmark": "longds",
                    "official_action": str(action_text),
                    "official_observations": observations,
                    "completion_reason": completion_reason,
                    "worker_budget_used": self.turn_step_count,
                    "worker_budget_limit": self.max_steps_per_turn,
                },
            )
        except Exception as exc:
            # Match DSGym's outer loop: surface the failure as a user observation and
            # allow the Worker another generation while respecting the same budget.
            error_text = (
                f"Error: Turn step {self.turn_step_count} failed: {exc}. "
                "Please try a different approach."
            )
            self.conversation.append({"role": "user", "content": error_text})
            self.environment.chat_history = copy.deepcopy(self.conversation)
            self._done = self.turn_step_count >= self.max_steps_per_turn
            if self._done:
                self._final_answer = ""
            return ReActStep(
                step_id=self.turn_step_count,
                action=AgentAction(
                    kind="tool",
                    reasoning="Worker/backend step failed before a usable action completed.",
                    tool_name="python",
                    arguments={"code": ""},
                ),
                observation=ToolResult(
                    "python",
                    False,
                    "",
                    data={"execution_attempted": False, "execution_succeeded": False},
                    error=f"{type(exc).__name__}: {exc}",
                ),
                done=self._done,
                metadata={
                    "benchmark": "longds",
                    "completion_reason": (
                        "budget_exhausted" if self._done else "step_error"
                    ),
                    "worker_budget_used": self.turn_step_count,
                    "worker_budget_limit": self.max_steps_per_turn,
                },
            )

    def inject_observation(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        metadata = metadata or {}
        if (
            metadata.get("stateguard") in {"light_repair", "heavy_repair"}
            and self.turn_step_count >= self.max_steps_per_turn
        ):
            raise RuntimeError(
                "cannot repair after the official LongDS Worker budget is exhausted"
            )
        rendered = (
            wrap_stateguard_observation(content)
            if metadata.get("stateguard")
            else content
        )
        self.conversation.append({"role": "user", "content": rendered})
        self.environment.chat_history = copy.deepcopy(self.conversation)
        # A repair after a submitted answer resumes the same turn without resetting
        # either DSGym's env counter or StateGuard's Worker budget counter.
        self._done = False
        self._final_answer = None

    def snapshot(self) -> LongDSWorkerSnapshot:
        return LongDSWorkerSnapshot(
            tuple(copy.deepcopy(self.conversation)),
            self.turn_step_count,
            self._done,
            self._final_answer,
            tuple(self.raw_responses),
            self.completed_step_count,
            self.turn_token_count,
        )

    def restore(self, snapshot: LongDSWorkerSnapshot) -> None:
        self.conversation = [dict(message) for message in copy.deepcopy(snapshot.conversation)]
        self.turn_step_count = snapshot.turn_step_count
        self._done = snapshot.done
        self._final_answer = snapshot.final_answer
        self.raw_responses = list(snapshot.raw_responses)
        self.completed_step_count = snapshot.completed_step_count
        self.turn_token_count = snapshot.turn_token_count
        self.environment.turns = snapshot.turn_step_count
        self.environment.chat_history = copy.deepcopy(self.conversation)

    def close(self) -> None:
        self.workspace.close()


def _require_execution_environment(
    environment: Any, metadata: dict[str, Any] | None
) -> None:
    """Fail before Worker/Manager execution when DSGym allocation silently failed.

    The upstream ``AllocatedCodeEnv.init`` currently reports
    ``container_allocated=True`` even inside its allocation exception path, so
    the live tool-group state is the authoritative signal when available.
    """
    tool_group = getattr(environment, "tool_group", None)
    if tool_group is not None and hasattr(tool_group, "allocated_container"):
        if tool_group.allocated_container is None:
            raise RuntimeError("LongDS code execution container was not allocated")
        return
    if metadata is not None and metadata.get("container_allocated") is False:
        raise RuntimeError("LongDS code execution container was not allocated")


def _coerce_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)):
        raise TypeError("LongDS Worker requires official native message objects")
    if not isinstance(value, Iterable):
        raise TypeError("LongDS Worker prompt must be an iterable of messages")
    messages = official_messages(value)
    if not messages:
        raise ValueError("LongDS Worker prompt cannot be empty")
    return messages
