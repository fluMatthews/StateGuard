from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from stateguard.core.models import TaskSpec
from stateguard.harness.blind_view import ManagerObservation, assert_blind
from stateguard.providers.base import ModelClient
from stateguard.runtime.tools import ToolRegistry
from stateguard.validation.models import ManagerDecision

from .react import ReActAgent, parse_json_object


MANAGER_SYSTEM_PROMPT = """You are the autonomous StateGuard manager, a GT-blind analytical-state controller.
Use only the task, worker trajectory, execution evidence, workspace manifest, and committed states supplied to you.
Ambiguity must not trigger repair. A repair requires high confidence, a violated constraint, and concrete evidence.
You may use the registered evidence tools, but you may not mutate worker state or commit state directly.
Return each ReAct step using the standard tool/final JSON envelope. The final answer must itself be the requested JSON decision.
"""


class StateManagerAgent(ReActAgent):
    """Long-lived ReAct controller whose terminal outputs are manager actions."""

    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry | None = None,
        *,
        max_steps_per_action: int = 8,
        system_prompt: str = MANAGER_SYSTEM_PROMPT,
    ) -> None:
        super().__init__(model, tools, system_prompt=system_prompt, max_steps=100000)
        self.max_steps_per_action = max_steps_per_action
        self.invocations: list[dict[str, Any]] = []

    def start_task(self, task: TaskSpec, state_index: list[dict[str, Any]]) -> None:
        payload = {
            "event_type": "TASK_INITIALIZATION",
            "task_id": task.id,
            "query": task.query,
            "context": task.context,
            "guidelines": task.guidelines,
            "data_files": task.data_files,
            "state_index": state_index,
        }
        assert_blind(payload)
        prompt = _load_prompt("manager_controller.txt") + "\n\nTASK:\n" + json.dumps(payload, ensure_ascii=False, default=str)
        if self.messages:
            self.inject_observation(
                "<task_unit_initialization>\n"
                + json.dumps(payload, ensure_ascii=False, default=str)
                + "\n</task_unit_initialization>"
            )
        else:
            self.start(prompt)

    def act(self, observation: ManagerObservation) -> ManagerDecision:
        payload = observation.to_dict()
        self.inject_observation(
            "<manager_observation>\n"
            + json.dumps(payload, ensure_ascii=False, default=str)
            + "\n</manager_observation>"
        )
        answer = self.run_for(self.max_steps_per_action)
        value = parse_json_object(answer)
        self.invocations.append(
            {"type": observation.event_type, "session": self.export_session(), "command": value}
        )
        return ManagerDecision.from_dict(value)


def _load_prompt(name: str) -> str:
    return files("stateguard.prompts").joinpath(name).read_text(encoding="utf-8")
