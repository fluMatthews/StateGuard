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
    """Compact Manager input for one review activation."""

    event_type: str
    task_id: str
    available_state_id: str
    untraced_steps: tuple[ReActStep, ...]
    pending_start_step: int | None
    pending_end_step: int | None
    current_draft: dict[str, Any] | None
    repair_attempts: int
    terminal_pending_review: bool = False
    worker_steps_remaining: int | None = None
    last_action_result: dict[str, Any] | None = None
    committed_state_index: tuple[dict[str, Any], ...] | None = None

    @property
    def worker_step(self) -> ReActStep | None:
        """Compatibility accessor; the duplicate step is not serialized."""
        return self.untraced_steps[-1] if self.untraced_steps else None

    @property
    def flow_policy(self) -> dict[str, Any]:
        """Compatibility accessor for dynamic terminal status only."""
        return {
            "terminal_pending_review": self.terminal_pending_review,
            "worker_steps_remaining": self.worker_steps_remaining,
        }

    def to_dict(self) -> dict[str, Any]:
        value = to_jsonable(self)
        value["untraced_steps"] = [
            _observed_step(step) for step in self.untraced_steps
        ]
        if self.committed_state_index is None:
            value.pop("committed_state_index", None)
        assert_blind(value)
        return value


def _observed_step(step: ReActStep) -> dict[str, Any]:
    """Project one Worker step down to what a review decision actually reads.

    ReActStep is the runtime audit record. It keeps the Worker's single raw
    response three times over: verbatim in raw_model_output, parsed into
    action.reasoning plus action.arguments, and copied again into
    metadata.official_action; the executed result appears both as observation
    and as metadata.official_observations. Sending all of them makes the
    Manager read the same code and the same stdout three times. The Worker's
    information is preserved here in exactly one form, and worker.jsonl keeps
    the complete step for audit and counterfactual replay.
    """
    action = step.action
    observed: dict[str, Any] = {"step_id": step.step_id, "done": step.done}
    action_view: dict[str, Any] = {"kind": action.kind}
    if action.reasoning:
        action_view["reasoning"] = action.reasoning
    if action.tool_name:
        action_view["tool_name"] = action.tool_name
    if action.arguments:
        action_view["arguments"] = to_jsonable(action.arguments)
    if action.answer is not None:
        action_view["answer"] = action.answer
    observed["action"] = action_view
    result = step.observation
    if result is not None:
        result_view: dict[str, Any] = {"ok": result.ok, "output": result.output}
        if result.error:
            result_view["error"] = result.error
        observed["observation"] = result_view
    return observed


class BlindViewBuilder:
    def build(
        self,
        *,
        task: TaskSpec,
        event_type: str,
        available_state_id: str,
        untraced_steps: tuple[ReActStep, ...],
        current_draft: dict[str, Any] | None,
        repair_attempts: int,
        terminal_pending_review: bool = False,
        worker_steps_remaining: int | None = None,
        last_action_result: dict[str, Any] | None = None,
        committed_state_index: tuple[dict[str, Any], ...] | None = None,
        **legacy: Any,
    ) -> ManagerObservation:
        legacy_flow = legacy.get("flow_policy")
        if isinstance(legacy_flow, dict):
            terminal_pending_review = bool(
                legacy_flow.get(
                    "terminal_pending_review",
                    legacy_flow.get(
                        "terminal_pending_state_required",
                        terminal_pending_review,
                    ),
                )
            )
            if worker_steps_remaining is None:
                raw_remaining = legacy_flow.get("worker_steps_remaining")
                if isinstance(raw_remaining, int):
                    worker_steps_remaining = raw_remaining
        assert_blind(task.metadata)
        request = ManagerObservation(
            event_type=event_type,
            task_id=task.id,
            available_state_id=available_state_id,
            untraced_steps=untraced_steps,
            pending_start_step=(
                untraced_steps[0].step_id if untraced_steps else None
            ),
            pending_end_step=(
                untraced_steps[-1].step_id if untraced_steps else None
            ),
            current_draft=current_draft,
            repair_attempts=repair_attempts,
            terminal_pending_review=terminal_pending_review,
            worker_steps_remaining=worker_steps_remaining,
            last_action_result=last_action_result,
            committed_state_index=committed_state_index,
        )
        request.to_dict()
        return request
