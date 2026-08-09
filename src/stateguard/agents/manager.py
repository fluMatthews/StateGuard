from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from stateguard.core.models import Message, TaskSpec
from stateguard.harness.blind_view import ManagerObservation, assert_blind
from stateguard.providers.base import ModelClient
from stateguard.runtime.tools import ToolRegistry
from stateguard.validation.models import ManagerDecision

from .react import ReActAgent, parse_json_object


MANAGER_SYSTEM_PROMPT = """
You are the autonomous StateGuard Manager supervising a worker agent. You observe the worker's analytical trajectory, maintain explicit analytical states, and intervene only when justified by evidence.

GOAL
- Preserve important intermediate variables, conclusions, and dependencies across a task.
- Detect clearly evidenced analytical errors before they propagate into later reasoning.
- Give local repair hints that help the worker reconsider an error without solving the task for it.
- Maintain verified analytical states and their semantic relations.

INFORMATION BOUNDARY
- Use only the task query, context, guidelines, data-file descriptions, worker trajectory, execution evidence, workspace evidence, current draft, and committed states supplied to you.
- Never use or request ground truth, reference answers, judge results.

AUTHORITY
- Autonomously decide whether an important analytical state has formed and whether evidence inspection, state operations, relation operations, or repair are warranted.
- You may use registered evidence tools for read-only inspection and isolated checks.
- You may request StateGuard control actions, but only the harness validates and executes state mutations, commits, rollbacks, worker resumption, and workspace cleanup.
- Never rewrite a committed state and never replace the worker by completing the analysis for it.

DECISION PRINCIPLES
- Trigger repair only when there is a clear violated constraint, concrete supporting evidence, and high confidence.
- Ambiguous or uncertain behavior must not trigger repair; pass and let the worker continue.
- Treat committed states as verified. Related-state review is upward-only and limited to one hop.
- Use evidence tools whenever execution facts or analytical checks require verification rather than assumption.

INTERACTION
- Follow the detailed controller protocol supplied at task initialization for lifecycle rules, state schemas, evidence tools, and StateGuard action formats.
- Return each internal evidence step with the standard ReAct tool envelope.
- Return exactly one StateGuard control action in the terminal ReAct final envelope when making a workflow decision.
"""


class StateManagerAgent(ReActAgent):
    """Long-lived ReAct controller whose terminal outputs are manager actions."""

    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry | None = None,
        *,
        max_steps_per_action: int = 8,
        max_context_chars: int | None = 120_000,
        system_prompt: str = MANAGER_SYSTEM_PROMPT,
    ) -> None:
        if max_context_chars is not None and max_context_chars < 1:
            raise ValueError("max_context_chars must be positive or None")
        super().__init__(model, tools, system_prompt=system_prompt, max_steps=100000)
        self.max_steps_per_action = max_steps_per_action
        self.max_context_chars = max_context_chars
        self.invocations: list[dict[str, Any]] = []
        self._lifecycle_prompt = ""

    def configure_lifecycle(self, lifecycle_prompt: str) -> None:
        """Bind benchmark lifecycle text before the first task unit starts."""
        lifecycle_prompt = lifecycle_prompt.strip()
        if self.messages and lifecycle_prompt != self._lifecycle_prompt:
            raise RuntimeError("manager lifecycle cannot change inside one task session")
        self._lifecycle_prompt = lifecycle_prompt

    def start_task(self, task: TaskSpec) -> None:
        payload = {
            "event_type": "TASK_INITIALIZATION",
            "task_id": task.id,
            "query": task.query,
            "context": task.context,
            "guidelines": task.guidelines,
            "data_files": task.data_files,
        }
        assert_blind(payload)
        controller = _load_prompt("manager_controller.txt").replace(
            "{{FLOW_LIFECYCLE}}",
            self._lifecycle_prompt or "Follow the lifecycle supplied in each manager observation.",
        )
        prompt = controller + "\n\nTASK:\n" + json.dumps(payload, ensure_ascii=False, default=str)
        if self.messages:
            self.inject_observation(
                "<task_unit_initialization>\n"
                + json.dumps(payload, ensure_ascii=False, default=str)
                + "\n</task_unit_initialization>",
                metadata={"manager_block_start": True, "event_type": "TASK_INITIALIZATION"},
            )
        else:
            self.start(prompt)

    def act(self, observation: ManagerObservation) -> ManagerDecision:
        payload = observation.to_dict()
        self.inject_observation(
            "<manager_observation>\n"
            + json.dumps(payload, ensure_ascii=False, default=str)
            + "\n</manager_observation>",
            metadata={"manager_block_start": True, "event_type": observation.event_type},
        )
        for attempt in range(2):
            try:
                answer = self.run_for(self.max_steps_per_action)
                value = parse_json_object(answer)
                decision = ManagerDecision.from_dict(value)
            except (KeyError, TypeError, ValueError) as exc:
                self.invocations.append(
                    {
                        "type": observation.event_type,
                        "session": self.export_session(),
                        "protocol_error": f"{type(exc).__name__}: {exc}",
                        "attempt": attempt + 1,
                    }
                )
                if attempt == 1:
                    raise
                self.inject_observation(
                    "<manager_protocol_error>\n"
                    "Your previous response was not executed because it did not match "
                    "the required ReAct/StateGuard JSON protocol.\n"
                    f"error_type: {type(exc).__name__}\n"
                    f"details: {exc}\n"
                    "Return one corrected action using the required JSON structure.\n"
                    "</manager_protocol_error>",
                    metadata={
                        "manager_block_start": False,
                        "event_type": observation.event_type,
                        "protocol_retry": True,
                    },
                )
                continue
            self.invocations.append(
                {
                    "type": observation.event_type,
                    "session": self.export_session(),
                    "command": value,
                    "attempt": attempt + 1,
                }
            )
            return decision
        raise AssertionError("manager protocol retry loop terminated unexpectedly")

    def _messages_for_model(self) -> list[Message]:
        """Pin initialization and retain the newest complete manager action blocks.

        The full logical manager session remains in ``self.session`` for artifacts and
        rollback. Only the model-facing payload is bounded. A block starts with one
        manager observation (or later task-unit initialization) and includes every
        following assistant tool call, tool result, and terminal decision, so a tool
        call is never separated from its result by truncation.
        """
        messages = list(self.session.messages)
        if self.max_context_chars is None or len(messages) <= 2:
            return messages

        pinned = messages[:2]
        body = messages[2:]
        blocks: list[list[Message]] = []
        current: list[Message] = []
        for message in body:
            if message.metadata.get("manager_block_start") and current:
                blocks.append(current)
                current = []
            current.append(message)
        if current:
            blocks.append(current)

        used = sum(self._message_size(message) for message in pinned)
        selected: list[list[Message]] = []
        for block in reversed(blocks):
            block_size = sum(self._message_size(message) for message in block)
            if selected and used + block_size > self.max_context_chars:
                break
            selected.append(block)
            used += block_size
            # Always preserve the newest complete block, even if it alone exceeds
            # the configured soft limit.
            if used > self.max_context_chars:
                break

        retained = [message for block in reversed(selected) for message in block]
        return pinned + retained

    @staticmethod
    def _message_size(message: Message) -> int:
        return len(message.content) + len(message.name or "")


def _load_prompt(name: str) -> str:
    return files("stateguard.prompts").joinpath(name).read_text(encoding="utf-8")
