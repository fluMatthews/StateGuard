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


_CHECK_TARGETS_BLOCK = "\n[[TRACK_CHECK_TARGETS]]\n"


@dataclass
class DACompWorkflow:
    """Single-query lifecycle over official DAComp Worker actions."""

    review_cadence: int = 5
    review_terminal_pending: bool = True
    hint_includes_variables: bool = False
    check_targets_name: str | None = None

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

    def _check_targets(self) -> str:
        """What this track's probe workspace can settle, or nothing.

        The two DE tracks ship different specifications -- Impl a data contract
        over an empty sql/, Evol a populated pipeline and the tables it already
        built -- so the parts worth recomputing differ. A track without a file
        renders the lifecycle exactly as before.
        """
        if not self.check_targets_name:
            return ""
        return (
            files("stateguard.adapters.dacomp")
            .joinpath(self.check_targets_name)
            .read_text(encoding="utf-8")
            .strip()
        )

    def _check_targets_block(self) -> str:
        """The section with its own blank lines, so a track without one leaves
        the surrounding text spaced exactly as it was."""
        targets = self._check_targets()
        return f"\n{targets}\n" if targets else ""

    def lifecycle_prompt(self) -> str:
        return (
            files("stateguard.adapters.dacomp")
            .joinpath("manager_lifecycle.txt")
            .read_text(encoding="utf-8")
            .replace("{{REVIEW_CADENCE}}", str(self.review_cadence))
            .replace(_CHECK_TARGETS_BLOCK, self._check_targets_block())
            .strip()
        )

    def manager_context(self) -> dict[str, Any]:
        return {
            "benchmark": "dacomp",
            "mode": "single_query",
            "review_cadence": self.review_cadence,
            "state_boundary": "manager-selected pending native step interval",
            "relation_timing": "select_once_after_state_body",
            "hint_policy": "empty_state_hint; repair_hint_only",
            "worker_budget_scope": "official accepted Worker actions; Manager actions excluded",
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


    def state_hint_includes_variables(self) -> bool:
        """DE-Impl only. A DE Worker learns the source data's defects by
        probing it -- that admin_id sits in the team_id column, that an email
        filter drops all but 7 of 3,004 rows -- and then writes SQL for dozens
        of steps before it needs them again. Those findings live in the state's
        variables and nowhere else the Worker can reach cheaply. DA reports and
        every other flow leave this off.
        """
        return self.hint_includes_variables

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
        if draft.source_interval is None:
            raise ValueError("write the selected DAComp interval before selecting relations")
        if draft.header.relations:
            raise ValueError("DAComp state header cannot contain provisional relations")
        if not draft.issue.strip():
            raise ValueError("DAComp state issue must be derived from the selected interval")
        if finalization.mode is not RelationFinalizationMode.SELECT:
            raise ValueError("DAComp relation finalization must use SELECT")
