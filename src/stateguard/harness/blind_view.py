from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stateguard.core.events import ReActStep
from stateguard.core.models import TaskSpec, to_jsonable


FORBIDDEN_KEYS = {
    "expected_answer",
    "gold",
    "gold_answer",
    "golden_answer",
    "ground_truth",
    "judge",
    "judge_reasoning",
    "reference_answer",
    "reward",
    "reward_spec",
}


def assert_blind(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).strip().lower() in FORBIDDEN_KEYS:
                raise ValueError(f"manager view contains forbidden key at {path}.{key}")
            assert_blind(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_blind(child, f"{path}[{index}]")


@dataclass(frozen=True)
class ManagerObservation:
    """One serialized input snapshot for the manager, not an agent or action."""

    event_type: str
    task_id: str
    query: str
    context: str
    guidelines: tuple[str, ...]
    flow_policy: dict[str, Any]
    available_state_id: str
    worker_step: ReActStep | None
    untraced_steps: tuple[ReActStep, ...]
    candidate_start_step: int | None
    candidate_end_step: int | None
    current_draft: dict[str, Any] | None
    workspace_manifest: dict[str, Any]
    repair_attempts: int
    last_action_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = to_jsonable(self)
        assert_blind(value)
        return value


class BlindViewBuilder:
    def build(
        self,
        *,
        task: TaskSpec,
        event_type: str,
        flow_policy: dict[str, Any],
        available_state_id: str,
        worker_step: ReActStep | None,
        untraced_steps: tuple[ReActStep, ...],
        current_draft: dict[str, Any] | None,
        workspace_manifest: dict[str, Any],
        repair_attempts: int,
        last_action_result: dict[str, Any] | None = None,
    ) -> ManagerObservation:
        assert_blind(task.metadata)
        request = ManagerObservation(
            event_type=event_type,
            task_id=task.id,
            query=task.query,
            context=task.context,
            guidelines=task.guidelines,
            flow_policy=flow_policy,
            available_state_id=available_state_id,
            worker_step=worker_step,
            untraced_steps=untraced_steps,
            candidate_start_step=(
                untraced_steps[0].step_id if untraced_steps else None
            ),
            candidate_end_step=(
                untraced_steps[-1].step_id if untraced_steps else None
            ),
            current_draft=current_draft,
            workspace_manifest=workspace_manifest,
            repair_attempts=repair_attempts,
            last_action_result=last_action_result,
        )
        request.to_dict()
        return request
