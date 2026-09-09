from __future__ import annotations

import json
from importlib.resources import files
from typing import Any
from urllib.error import HTTPError, URLError

from stateguard.core.models import Message, TaskSpec
from stateguard.harness.blind_view import ManagerObservation, assert_blind
from stateguard.providers.base import ModelClient
from stateguard.runtime.tools import ToolRegistry
from stateguard.sft.context import (
    DEFAULT_MANAGER_CONTEXT_CHARS,
    DEFAULT_MANAGER_OUTPUT_RESERVE_CHARS,
    select_manager_context,
)
from stateguard.validation.models import ManagerDecision

from .react import ReActAgent, parse_json_object

MANAGER_SYSTEM_PROMPT = """
You are a State Manager supervising a Worker agent.

1. ROLE
- Observe the Worker trajectory and maintain explicit analytical states and relations.
- Detect clear analytical errors, localize their evidence, and request limited repair.
- Protect the Worker from unnecessary intervention; do not solve the task for it.

2. INFORMATION BOUNDARY
- Use only the task information, Worker trace, execution evidence, workspace evidence,
  current draft, and committed states supplied through the runtime.
- Never use or request ground truth, reference answers, or judge results.

3. DECISION POLICY
- Follow the controller protocol and the active flow-specific lifecycle.
- Use Tool actions mainly for checking and verifying worker's trajectory, 
  especially the evidence which will be relied on by later states.
- When a check that produces a concrete violation with evidence, you must to do REPAIR.
  Repair only a concrete violation supported by clear evidence; otherwise pass.
  When the evidence is ambiguous, pass it as well,
- Treat committed states as locked and never modify them.

4. AUTHORITY
- There are exactly two different action classes.
- Tool actions obtain missing evidence through FunctionTools. They use type=tool,
  return a tool result, and do not end the current Manager activation.
- Control actions make StateGuard lifecycle decisions. They use type=control,
  are plain assistant JSON rather than tool calls, and end the current activation.
- Never call a control action as a tool, and never combine a tool action with a
  control action in one response.
"""

FLOW_STATE_ACTIONS_MARKER = "[[FLOW_STATE_ACTIONS]]"

_RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}


# OPEN_STATE is the only mandatory turn-start decision and it gets no lifecycle
# retry, so a correction that just repeats "use the required JSON structure"
# wastes the single attempt. Show the exact structure instead.
_OPEN_STATE_TEMPLATE = (
    '{"type":"control","reasoning":"...","answer":{"action":"OPEN_STATE",'
    '"state_header":{"issue":"...","constraints":[{"text":"..."}],'
    '"relations":[{"type":"...","related_state_id":"..."}]}}}'
)


def _is_retryable_transport_error(exc: Exception) -> bool:
    """Return whether a failed model request may be retried once safely."""
    if isinstance(exc, HTTPError):
        return exc.code in _RETRYABLE_HTTP_STATUS
    return isinstance(exc, (TimeoutError, ConnectionError, URLError))



class StateManagerAgent(ReActAgent):
    """Long-lived ReAct controller whose terminal outputs are manager actions."""

    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry | None = None,
        *,
        max_steps_per_action: int = 8,
        max_context_chars: int | None = DEFAULT_MANAGER_CONTEXT_CHARS,
        reserved_output_chars: int = DEFAULT_MANAGER_OUTPUT_RESERVE_CHARS,
        system_prompt: str = MANAGER_SYSTEM_PROMPT,
    ) -> None:
        if max_context_chars is not None and max_context_chars < 1:
            raise ValueError("max_context_chars must be positive or None")
        if reserved_output_chars < 0:
            raise ValueError("reserved_output_chars cannot be negative")
        super().__init__(model, tools, system_prompt=system_prompt, max_steps=100000)
        self.max_steps_per_action = max_steps_per_action
        self.max_context_chars = max_context_chars
        self.reserved_output_chars = reserved_output_chars
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
            "query": task.metadata.get("manager_query") or task.query,
            "context": task.context,
            "guidelines": task.guidelines,
            "data_files": task.data_files,
        }
        assert_blind(payload)
        controller = render_manager_controller(
            self._lifecycle_prompt
            or "Follow the lifecycle supplied in each manager observation."
        )
        prompt = controller + "\n\nTASK:\n" + json.dumps(payload, ensure_ascii=False, default=str)
        if self.messages:
            self.inject_observation(
                "<task_unit_initialization>\n"
                + json.dumps(payload, ensure_ascii=False, default=str)
                + "\n</task_unit_initialization>",
                metadata={"manager_block_start": True, "event_type": "TASK_INITIALIZATION", "task_id": task.id},
            )
        else:
            self.start(prompt)

    def act(self, observation: ManagerObservation) -> ManagerDecision:
        payload = observation.to_dict()
        self.inject_observation(
            "<manager_observation>\n"
            + json.dumps(payload, ensure_ascii=False, default=str)
            + "\n</manager_observation>",
            metadata={"manager_block_start": True, "event_type": observation.event_type, "task_id": observation.task_id},
        )
        for attempt in range(2):
            protocol_attempt_start_step = self.session.step_count
            transport_retries_remaining = 1
            while True:
                try:
                    used_steps = self.session.step_count - protocol_attempt_start_step
                    remaining_steps = self.max_steps_per_action - used_steps
                    if remaining_steps < 1:
                        raise RuntimeError(
                            f"agent activation exceeded {self.max_steps_per_action} ReAct steps"
                        )
                    answer = self.run_for(remaining_steps)
                    break
                except Exception as exc:
                    retryable = _is_retryable_transport_error(exc)
                    will_retry = transport_retries_remaining > 0
                    if retryable:
                        self.invocations.append(
                            {
                                "type": observation.event_type,
                                "session": self.export_session(),
                                "transport_error": f"{type(exc).__name__}: {exc}",
                                "will_retry": will_retry,
                            }
                        )
                        if will_retry:
                            transport_retries_remaining -= 1
                            continue
                    # Preserve the existing final runtime-failure record. The
                    # harness catches this exception and remains fail-open.
                    self.invocations.append(
                        {
                            "type": observation.event_type,
                            "session": self.export_session(),
                            "runtime_error": f"{type(exc).__name__}: {exc}",
                            "attempt": attempt + 1,
                        }
                    )
                    raise
            try:
                terminal = parse_json_object(self.session.messages[-1].content)
                if str(terminal.get("type", "")).lower() != "control":
                    raise ValueError(
                        "Manager terminal actions must use type=control; "
                        "type=final is reserved for Worker answers"
                    )
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
                correction = (
                    "<manager_protocol_error>\n"
                    "Your previous response was not executed because it did not match "
                    "the required ReAct/StateGuard JSON protocol.\n"
                    f"error_type: {type(exc).__name__}\n"
                    f"details: {exc}\n"
                )
                if observation.current_draft is None:
                    correction += (
                        "This review event expects OPEN_STATE. Return exactly:\n"
                        f"{_OPEN_STATE_TEMPLATE}\n"
                    )
                else:
                    correction += (
                        "Return one corrected Tool or control action using the "
                        "required JSON structure.\n"
                    )
                correction += "</manager_protocol_error>"
                self.inject_observation(
                    correction,
                    metadata={
                        "manager_block_start": False,
                        "event_type": observation.event_type,
                        "protocol_retry": True,
                    },
                )
                continue
            except Exception as exc:
                # Preserve the untouched model/tool trajectory even when an
                # activation exhausts its budget or fails before a valid control
                # action is produced. The harness remains fail-open for Worker.
                self.invocations.append(
                    {
                        "type": observation.event_type,
                        "session": self.export_session(),
                        "runtime_error": f"{type(exc).__name__}: {exc}",
                        "attempt": attempt + 1,
                    }
                )
                raise
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
        return select_manager_context(
            self.session.messages,
            self.max_context_chars,
            self.reserved_output_chars,
        )


def render_manager_controller(lifecycle_prompt: str) -> str:
    """Place lifecycle rules and flow-specific state actions in their sections."""
    lifecycle, marker, state_actions = lifecycle_prompt.partition(
        FLOW_STATE_ACTIONS_MARKER
    )
    if not marker:
        state_actions = (
            "5.2 OPEN_STATE\n\n"
            "Use the header required by this lifecycle; the harness assigns id.\n\n"
            "5.3 UPDATE_STATE\n\n"
            "Write accepted Worker trace; the harness assigns variable versions.\n\n"
            "5.4 FINALIZE_RELATIONS\n\n"
            "Use only the relation mode and fields allowed by this lifecycle."
        )
    controller = _load_prompt("manager_controller.txt")
    return (
        controller.replace("{{FLOW_LIFECYCLE}}", lifecycle.strip())
        .replace("{{FLOW_STATE_ACTIONS}}", state_actions.strip())
    )


def _load_prompt(name: str) -> str:
    return files("stateguard.prompts").joinpath(name).read_text(encoding="utf-8")
