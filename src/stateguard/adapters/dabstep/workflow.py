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
class DABstepWorkflow:
    """Single-query lifecycle over official DABstep code-action steps."""

    review_cadence: int = 3
    review_terminal_pending: bool = True

    def __post_init__(self) -> None:
        if self.review_cadence < 1:
            raise ValueError("review_cadence must be positive")

    def start(self, task: TaskSpec) -> None:
        del task

    def review_before_worker(self) -> bool:
        return False

    def prepare_worker(self, worker: Agent, task: TaskSpec, workspace: Workspace) -> None:
        del workspace
        # The DABstep facade uses TaskSpec to reconstruct the exact official prompt.
        worker.start(task)

    def should_review(self, step: ReActStep, steps_since_review: int) -> bool:
        return step.done or steps_since_review >= self.review_cadence

    def review_event_type(self, step: ReActStep) -> str:
        del step
        return "STEP_WINDOW"

    def lifecycle_prompt(self) -> str:
        return (
            files("stateguard.adapters.dabstep")
            .joinpath("manager_lifecycle.txt")
            .read_text(encoding="utf-8")
            .replace("{{REVIEW_CADENCE}}", str(self.review_cadence))
            .strip()
        )

    def manager_context(self) -> dict[str, Any]:
        return {
            "benchmark": "dabstep",
            "mode": "single_query",
            "review_cadence": self.review_cadence,
            "state_boundary": "manager-selected pending native step interval",
            "relation_timing": "select_once_after_state_body",
            "hint_policy": "empty_state_hint; repair_hint_only",
            "worker_budget_scope": (
                "official DABstep code-action steps; Manager actions excluded"
            ),
            "review_terminal_pending": self.review_terminal_pending,
        }

    def hint_state_ids(self, provisional_relation_ids: tuple[str, ...]) -> tuple[str, ...]:
        del provisional_relation_ids
        return ()

    def resumes_with_state_summary(self) -> bool:
        """This flow opens states on its own schedule, so a Worker can run a
        long stretch with no state hint at all. The harness sends the newest
        committed states when it resumes, and only when the store changed.
        """
        return True

    def validate_state_open(
        self, header: StateHeader, untraced_steps: tuple[ReActStep, ...]
    ) -> None:
        if not untraced_steps:
            raise ValueError("DABstep opens a state only after pending Worker steps exist")
        if header.relations:
            raise ValueError("DABstep relations are selected after writing the state body")
        if header.issue.strip():
            raise ValueError(
                "DABstep OPEN_STATE contains only id and query constraints; derive "
                "the issue from the selected interval in UPDATE_STATE"
            )

    def validate_relation_finalization(
        self,
        *,
        draft: StateDraft,
        finalization: RelationFinalization,
        untraced_steps: tuple[ReActStep, ...],
    ) -> None:
        del untraced_steps
        if draft.source_interval is None:
            raise ValueError("write the selected DABstep interval before relations")
        if draft.header.relations:
            raise ValueError("DABstep OPEN_STATE cannot contain provisional relations")
        if not draft.issue.strip():
            raise ValueError("DABstep state issue must come from the selected interval")
        if finalization.mode is not RelationFinalizationMode.SELECT:
            raise ValueError("DABstep relations must be finalized posthoc with SELECT")
