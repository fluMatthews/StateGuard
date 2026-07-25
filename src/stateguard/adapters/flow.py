from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from stateguard.core.events import ReActStep
from stateguard.agents.base import Agent
from stateguard.core.models import TaskSpec
from stateguard.state.draft import (
    RelationFinalization,
    RelationFinalizationMode,
    StateDraft,
    StateHeader,
)


class RelationTiming(str, Enum):
    QUERY_FIRST = "query_first"
    SEGMENT_COMPLETE = "segment_complete"


class FlowAdapter(Protocol):
    """Task-timing semantics consumed by the common StateGuard harness."""

    def start(self, task: TaskSpec) -> None: ...

    def prepare_worker(self, worker: Agent, prompt: str) -> None: ...

    def should_review(self, step: ReActStep, steps_since_review: int) -> bool: ...

    def review_event_type(self, step: ReActStep) -> str: ...

    def manager_context(self) -> dict[str, Any]: ...

    def validate_state_open(
        self,
        header: StateHeader,
        untraced_steps: tuple[ReActStep, ...],
    ) -> None: ...

    def hint_state_ids(self, provisional_relation_ids: tuple[str, ...]) -> tuple[str, ...]: ...

    def validate_relation_finalization(
        self,
        *,
        draft: StateDraft,
        finalization: RelationFinalization,
        untraced_steps: tuple[ReActStep, ...],
    ) -> None: ...


@dataclass
class TurnFlowAdapter:
    """One analytical state per task/turn, initialized from its query."""

    turn_end_metadata_key: str = "turn_end"

    def start(self, task: TaskSpec) -> None:
        del task

    def prepare_worker(self, worker: Agent, prompt: str) -> None:
        if worker.messages:
            worker.continue_turn(prompt)
        else:
            worker.start(prompt)

    def hint_state_ids(self, provisional_relation_ids: tuple[str, ...]) -> tuple[str, ...]:
        return provisional_relation_ids

    def validate_relation_finalization(
        self,
        *,
        draft: StateDraft,
        finalization: RelationFinalization,
        untraced_steps: tuple[ReActStep, ...],
    ) -> None:
        if not draft.traced_step_ids or untraced_steps:
            raise ValueError(
                "turn flow must trace the completed turn into current state before "
                "FINALIZE_RELATIONS"
            )
        if finalization.mode is RelationFinalizationMode.CONFIRM:
            if set(finalization.relations) != set(draft.header.relations):
                raise ValueError("confirm must preserve the provisional LongDS relations")
            return
        if finalization.mode is RelationFinalizationMode.RESELECT:
            if set(finalization.relations) == set(draft.header.relations):
                raise ValueError("reselect must replace the conflicting provisional relations")
            return
        raise ValueError("LongDS relation finalization must use confirm or reselect")

    def should_review(self, step: ReActStep, steps_since_review: int) -> bool:
        del steps_since_review
        return bool(
            step.done
            or step.metadata.get(self.turn_end_metadata_key)
            or step.action.metadata.get(self.turn_end_metadata_key)
        )

    def review_event_type(self, step: ReActStep) -> str:
        del step
        return "TURN_END"

    def manager_context(self) -> dict[str, Any]:
        return {
            "mode": "turn",
            "state_boundary": "one task/turn",
            "relation_timing": RelationTiming.QUERY_FIRST.value,
            "relation_instruction": (
                "At turn start, read the current query and the contents of stored_states, then "
                "write provisional exact related state IDs before tracing. After the current "
                "state is fully written and checked, validate it against those selected states. "
                "Confirm by default; only on explicit conflict use the shared current-state plus "
                "full-store selection procedure to replace them."
            ),
            "hint_policy": "provisional_relation_states",
            "relation_finalization": "confirm_or_reselect_on_explicit_conflict",
        }

    def validate_state_open(
        self,
        header: StateHeader,
        untraced_steps: tuple[ReActStep, ...],
    ) -> None:
        if untraced_steps:
            raise ValueError(
                "turn flow requires OPEN_STATE at turn start before worker trace is produced"
            )
        if not header.relations:
            raise ValueError("turn flow requires query-first provisional relations")


@dataclass
class FixedStepFlowAdapter:
    """Review fixed worker-step windows; the manager may or may not form state."""

    window_size: int = 5

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive")

    def start(self, task: TaskSpec) -> None:
        del task

    def prepare_worker(self, worker: Agent, prompt: str) -> None:
        worker.start(prompt)

    def hint_state_ids(self, provisional_relation_ids: tuple[str, ...]) -> tuple[str, ...]:
        del provisional_relation_ids
        return ()

    def validate_relation_finalization(
        self,
        *,
        draft: StateDraft,
        finalization: RelationFinalization,
        untraced_steps: tuple[ReActStep, ...],
    ) -> None:
        if not draft.traced_step_ids or untraced_steps:
            raise ValueError(
                "fixed-step flow must write the complete current state before selecting relations"
            )
        if draft.header.relations:
            raise ValueError("fixed-step state header must not contain provisional relations")
        if finalization.mode is not RelationFinalizationMode.SELECT:
            raise ValueError("fixed-step relation finalization must use select")

    def should_review(self, step: ReActStep, steps_since_review: int) -> bool:
        return step.done or steps_since_review >= self.window_size

    def review_event_type(self, step: ReActStep) -> str:
        del step
        return "STEP_WINDOW"

    def manager_context(self) -> dict[str, Any]:
        return {
            "mode": "fixed_step",
            "review_window": self.window_size,
            "state_boundary": "manager decides at each review point",
            "relation_timing": RelationTiming.SEGMENT_COMPLETE.value,
            "relation_instruction": (
                "After a completed segment forms a state, OPEN_STATE with no relations, write the "
                "complete current state, then use that current state and all stored_states to "
                "select the final exact five-type relation IDs once."
            ),
            "hint_policy": "empty",
            "relation_finalization": "select_once_after_current_state_is_written",
        }

    def validate_state_open(
        self,
        header: StateHeader,
        untraced_steps: tuple[ReActStep, ...],
    ) -> None:
        if not untraced_steps:
            raise ValueError(
                "fixed-step flow requires segment output before manager opens a state"
            )
        if header.relations:
            raise ValueError(
                "fixed-step flow defers relation selection until current state is written"
            )
