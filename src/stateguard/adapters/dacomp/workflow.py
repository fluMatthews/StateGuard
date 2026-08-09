from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from stateguard.agents.base import Agent
from stateguard.core.events import ReActStep
from stateguard.core.models import TaskSpec
from stateguard.runtime.workspace import Workspace
from stateguard.state.draft import (
    RelationFinalization,
    RelationFinalizationMode,
    StateDraft,
    StateHeader,
)


@dataclass
class DACompWorkflow:
    """StateGuard lifecycle for one DAComp single-query task.

    Five accepted native Worker actions are a pause cadence, never a state boundary.
    The Manager decides whether an important state exists and which pending interval
    belongs to it. Relations are selected only after the state body has been written.
    """

    review_cadence: int = 5

    def __post_init__(self) -> None:
        if self.review_cadence < 1:
            raise ValueError("review_cadence must be positive")

    def start(self, task: TaskSpec) -> None:
        del task

    def review_before_worker(self) -> bool:
        return False

    def prepare_worker(self, worker: Agent, task: TaskSpec, workspace: Workspace) -> None:
        del workspace
        worker.start(task.query)

    def should_review(self, step: ReActStep, steps_since_review: int) -> bool:
        return step.done or steps_since_review >= self.review_cadence

    def review_event_type(self, step: ReActStep) -> str:
        del step
        return "STEP_WINDOW"

    def lifecycle_prompt(self) -> str:
        return (
            files("stateguard.adapters.dacomp")
            .joinpath("manager_lifecycle.txt")
            .read_text(encoding="utf-8")
            .replace("{{REVIEW_CADENCE}}", str(self.review_cadence))
            .strip()
        )

    def manager_context(self) -> dict[str, Any]:
        return {
            "benchmark": "dacomp",
            "mode": "single_query",
            "review_cadence": self.review_cadence,
            "state_boundary": "manager-selected pending action interval",
            "relation_timing": "select_once_after_state_body",
            "hint_policy": "empty_state_hint; repair_hint_only",
            "worker_budget_scope": "official accepted Worker actions; Manager actions excluded",
        }

    def hint_state_ids(self, provisional_relation_ids: tuple[str, ...]) -> tuple[str, ...]:
        del provisional_relation_ids
        return ()

    def validate_state_open(
        self, header: StateHeader, untraced_steps: tuple[ReActStep, ...]
    ) -> None:
        if not untraced_steps:
            raise ValueError("DAComp opens a state only after observing pending Worker actions")
        if header.relations:
            raise ValueError("DAComp relations are selected after the state body is written")
        if header.issue.strip():
            raise ValueError(
                "DAComp OPEN_STATE contains only id and query constraints; write the "
                "interval-defined issue in UPDATE_STATE"
            )

    def validate_relation_finalization(
        self,
        *,
        draft: StateDraft,
        finalization: RelationFinalization,
        untraced_steps: tuple[ReActStep, ...],
    ) -> None:
        del untraced_steps
        if not draft.traced_step_ids:
            raise ValueError("write the selected DAComp interval before selecting relations")
        if draft.header.relations:
            raise ValueError("DAComp state header cannot contain provisional relations")
        if not draft.issue.strip():
            raise ValueError("DAComp state issue must be derived from the selected interval")
        if finalization.mode is not RelationFinalizationMode.SELECT:
            raise ValueError("DAComp relation finalization must use SELECT")
