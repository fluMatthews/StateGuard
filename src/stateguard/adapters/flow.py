from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from stateguard.core.events import ReActStep
from stateguard.agents.base import Agent
from stateguard.core.models import TaskSpec
from stateguard.runtime.workspace import Workspace
from stateguard.state.draft import (
    RelationFinalization,
    RelationFinalizationMode,
    StateDraft,
    StateHeader,
)

from .protocol import BenchmarkWorkflow


class RelationTiming(str, Enum):
    QUERY_FIRST = "query_first"
    SEGMENT_COMPLETE = "segment_complete"


# Backward-compatible name. New integrations implement one benchmark-specific
# BenchmarkWorkflow inside their own adapter package.
FlowAdapter = BenchmarkWorkflow


@dataclass
class TurnFlowAdapter:
    """Deprecated reference workflow; real benchmarks own a dedicated adapter."""

    turn_end_metadata_key: str = "turn_end"

    def start(self, task: TaskSpec) -> None:
        del task

    def review_before_worker(self) -> bool:
        return True

    def prepare_worker(
        self, worker: Agent, task: TaskSpec, workspace: Workspace
    ) -> None:
        staged_files = workspace.stage_data_files(task.data_files) if task.data_files else {}
        prompt = task.initial_prompt()
        if staged_files:
            import json

            prompt += (
                "\n\n<workspace_data_files>\n"
                + json.dumps(staged_files, ensure_ascii=False, indent=2)
                + "\n</workspace_data_files>\n"
                "Use the persistent python tool and the data_files mapping to read these files."
            )
        if worker.messages:
            worker.continue_turn(prompt)
        else:
            worker.start(prompt)

    def lifecycle_prompt(self) -> str:
        return (
            "One public task unit is one analytical turn. At turn start, open the "
            "state and provisionally select relations from the query and committed "
            "store before resuming the worker. Review after the completed turn, write "
            "and check the state, then confirm relations or reselect only on explicit "
            "conflict before commit. The harness binds the complete turn automatically, "
            "including repair history; the Manager never enumerates step IDs."
        )

    def hint_state_ids(self, provisional_relation_ids: tuple[str, ...]) -> tuple[str, ...]:
        return provisional_relation_ids

    def validate_relation_finalization(
        self,
        *,
        draft: StateDraft,
        finalization: RelationFinalization,
        untraced_steps: tuple[ReActStep, ...],
    ) -> None:
        if draft.source_interval is None or untraced_steps:
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
                "At turn start, use the harness-supplied compact state index with the current "
                "query, then write provisional relation IDs before tracing. After the current "
                "state is fully written and checked, use the automatically refreshed compact "
                "index and call load_state only for exact related states needed to verify or "
                "reselect a relation on explicit conflict."
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
        if not header.issue.strip():
            raise ValueError("turn flow requires the query-defined issue in the state header")


@dataclass
class FixedStepFlowAdapter:
    """Deprecated reference workflow; real benchmarks own a dedicated adapter."""

    window_size: int = 5

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive")

    def start(self, task: TaskSpec) -> None:
        del task

    def review_before_worker(self) -> bool:
        return True

    def prepare_worker(
        self, worker: Agent, task: TaskSpec, workspace: Workspace
    ) -> None:
        staged_files = workspace.stage_data_files(task.data_files) if task.data_files else {}
        prompt = task.initial_prompt()
        if staged_files:
            import json

            prompt += (
                "\n\n<workspace_data_files>\n"
                + json.dumps(staged_files, ensure_ascii=False, indent=2)
                + "\n</workspace_data_files>"
            )
        worker.start(prompt)

    def lifecycle_prompt(self) -> str:
        return (
            "Pause at the benchmark review cadence. A pause is only an observation "
            "point: first decide whether pending worker steps form an important state. "
            "Only then open, write, check, relate, and commit that selected interval."
        )

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
        if draft.source_interval is None:
            raise ValueError(
                "fixed-step flow must write the selected current-state interval before "
                "selecting relations"
            )
        if draft.header.relations:
            raise ValueError("fixed-step state header must not contain provisional relations")
        if not draft.issue.strip():
            raise ValueError(
                "fixed-step flow must write the interval-defined issue in UPDATE_STATE"
            )
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
                "The review window is only a pause cadence. If pending steps form a state, "
                "OPEN_STATE with no relations, then UPDATE_STATE with one inclusive "
                "source_interval. Its start and end may be any observed pending steps and need "
                "not align with the review window. Later steps remain pending; after repair "
                "preserve the interval start and extend only its end. "
                "After writing and checking the selected current state, use the harness-supplied "
                "compact id/issue/conclusions index to select final five-type relation IDs once."
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
        if header.issue.strip():
            raise ValueError(
                "fixed-step OPEN_STATE may contain only ID and query constraints; "
                "write the interval-defined issue in UPDATE_STATE"
            )
