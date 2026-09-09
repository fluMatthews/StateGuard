from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


MANIFEST_VERSION = 1


@dataclass(frozen=True)
class WorkerReplayCall:
    call_index: int
    unit_id: str
    step_id: int
    input_hash: str
    response: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkerReplayCall":
        return cls(
            int(value["call_index"]),
            str(value["unit_id"]),
            int(value["step_id"]),
            str(value["input_hash"]),
            str(value["response"]),
        )


@dataclass(frozen=True)
class ManagerReplayCall:
    call_index: int
    task_id: str
    event_type: str
    input_hash: str
    response: str
    reasoning: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ManagerReplayCall":
        return cls(
            int(value["call_index"]),
            str(value.get("task_id", "")),
            str(value.get("event_type", "")),
            str(value["input_hash"]),
            str(value["response"]),
            str(value.get("reasoning", "")),
        )


@dataclass(frozen=True)
class ReplayManifest:
    parent_run: str
    source: str
    mode: str
    task_id: str
    worker_calls: tuple[WorkerReplayCall, ...]
    manager_calls: tuple[ManagerReplayCall, ...]
    parent_files: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.version != MANIFEST_VERSION:
            raise ValueError(f"unsupported replay manifest version: {self.version}")
        _require_sequential(self.worker_calls, "Worker")
        _require_sequential(self.manager_calls, "Manager")

    def worker_call(self, unit_id: str, step_id: int) -> WorkerReplayCall:
        matches = [
            call
            for call in self.worker_calls
            if call.unit_id == unit_id and call.step_id == step_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one Worker call for {unit_id} step {step_id}; "
                f"found {len(matches)}"
            )
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReplayManifest":
        return cls(
            parent_run=str(value["parent_run"]),
            source=str(value["source"]),
            mode=str(value["mode"]),
            task_id=str(value["task_id"]),
            worker_calls=tuple(
                WorkerReplayCall.from_dict(row) for row in value["worker_calls"]
            ),
            manager_calls=tuple(
                ManagerReplayCall.from_dict(row) for row in value["manager_calls"]
            ),
            parent_files={str(key): str(item) for key, item in value["parent_files"].items()},
            metadata=dict(value.get("metadata") or {}),
            version=int(value.get("version", MANIFEST_VERSION)),
        )


@dataclass(frozen=True)
class InterventionPlan:
    """One Worker-only branch point plus optional repair demonstrations.

    ``replacement_action`` replaces the clean Worker response at the selected
    call. ``forced_manager_responses`` supplies a complete Manager action
    sequence ending in REPAIR after the branch point; when it is empty, the
    Manager remains fully live. ``forced_repair_responses`` supplies Worker
    responses only after a Manager REPAIR is accepted by the harness. The
    optional post-repair Manager sequence keeps the demonstration active until a
    corrected state is committed; omitting it preserves the original immediate
    live-release behavior. Concrete mutation operators and demonstration builders
    live outside this protocol.
    """

    intervention_id: str
    target_unit_id: str
    target_step_id: int
    replacement_action: str
    forced_repair_responses: tuple[str, ...] = ()
    forced_manager_responses: tuple[str, ...] = ()
    forced_post_repair_manager_responses: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.intervention_id.strip():
            raise ValueError("intervention_id must be non-empty")
        if not self.target_unit_id.strip() or self.target_step_id < 1:
            raise ValueError("a valid Worker unit and step are required")
        if not self.replacement_action.strip():
            raise ValueError("replacement_action must be non-empty")
        if any(not response.strip() for response in self.forced_manager_responses):
            raise ValueError("forced Manager responses must be non-empty")
        if any(not response.strip() for response in self.forced_repair_responses):
            raise ValueError("forced Worker repair responses must be non-empty")
        if any(
            not response.strip()
            for response in self.forced_post_repair_manager_responses
        ):
            raise ValueError("forced post-repair Manager responses must be non-empty")
        if self.forced_post_repair_manager_responses and not (
            self.forced_manager_responses and self.forced_repair_responses
        ):
            raise ValueError(
                "post-repair Manager responses require both a forced Manager "
                "REPAIR sequence and a forced Worker repair response"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InterventionPlan":
        return cls(
            intervention_id=str(value["intervention_id"]),
            target_unit_id=str(value["target_unit_id"]),
            target_step_id=int(value["target_step_id"]),
            replacement_action=str(value["replacement_action"]),
            forced_manager_responses=tuple(
                str(item) for item in value.get("forced_manager_responses", ())
            ),
            forced_repair_responses=tuple(
                str(item) for item in value.get("forced_repair_responses", ())
            ),
            forced_post_repair_manager_responses=tuple(
                str(item)
                for item in value.get("forced_post_repair_manager_responses", ())
            ),
            metadata=dict(value.get("metadata") or {}),
        )


def _require_sequential(calls: tuple[Any, ...], label: str) -> None:
    indices = [int(call.call_index) for call in calls]
    expected = list(range(1, len(calls) + 1))
    if indices != expected:
        raise ValueError(f"{label} replay calls are not sequential: {indices}")
