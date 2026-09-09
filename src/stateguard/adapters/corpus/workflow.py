from __future__ import annotations

from pathlib import Path
from typing import Any

from stateguard.adapters.longds.prompts import render_turn_messages
from stateguard.adapters.longds.workflow import LongDSWorkflow
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

from .dataset import CorpusPublicUnit


_SINGLE_QUERY_LIFECYCLE = (
    (Path(__file__).resolve().parents[1] / "dabstep" / "manager_lifecycle.txt")
    .read_text(encoding="utf-8")
    .replace("SINGLE-QUERY LIFECYCLE", "SINGLE-QUERY CORPUS LIFECYCLE", 1)
    .replace(
        "accepted official Worker action steps",
        "accepted Worker action steps",
    )
)





class CorpusMultiTurnWorkflow(LongDSWorkflow):
    """LongDS lifecycle applied to an explicitly segmented corpus task."""

    def prepare_worker(
        self, worker: Agent, task: TaskSpec, workspace: Workspace
    ) -> None:
        del workspace
        unit = self._turn(task)
        messages = render_turn_messages(
            system_prompt_template=self.system_prompt_template,
            data_root=unit.data_root,
            context=unit.context,
            question=unit.query,
            first_turn=not bool(worker.messages),
        )
        if worker.messages:
            worker.continue_turn(messages)
        else:
            worker.start(messages)

    def manager_context(self) -> dict[str, Any]:
        return {
            "benchmark": "corpus",
            "mode": "turn",
            "state_boundary": "one complete corpus turn",
            "relation_timing": "query_first_then_confirm_or_reselect",
            "hint_policy": "provisional_relation_states",
            "pause_policy": "after Worker turn completion only",
            "worker_budget_scope": "per_turn; Manager actions do not consume it",
        }

    def lifecycle_prompt(self) -> str:
        return super().lifecycle_prompt().replace(
            "LONGDS TURN LIFECYCLE", "CORPUS MULTI-TURN LIFECYCLE"
        )


class CorpusSingleQueryWorkflow:
    """DABstep-equivalent state formation over one generic data-analysis query."""

    def __init__(
        self,
        unit: CorpusPublicUnit,
        system_prompt_template: str,
        review_cadence: int = 3,
    ) -> None:
        if review_cadence < 1:
            raise ValueError("review_cadence must be positive")
        self.review_cadence = review_cadence
        self.review_terminal_pending = True
        self.unit = unit
        self.system_prompt_template = system_prompt_template

    def start(self, task: TaskSpec) -> None:
        del task

    def review_before_worker(self) -> bool:
        return False

    def prepare_worker(
        self, worker: Agent, task: TaskSpec, workspace: Workspace
    ) -> None:
        del workspace
        if task.id != self.unit.unit_id:
            raise ValueError("single-query TaskSpec does not match the bound corpus unit")
        messages = render_turn_messages(
            system_prompt_template=self.system_prompt_template,
            data_root=self.unit.data_root,
            context=self.unit.context,
            question=self.unit.query,
            first_turn=True,
        )
        worker.start(messages)

    def should_review(self, step: ReActStep, steps_since_review: int) -> bool:
        return step.done or steps_since_review >= self.review_cadence

    def review_event_type(self, step: ReActStep) -> str:
        del step
        return "STEP_WINDOW"

    def manager_context(self) -> dict[str, Any]:
        return {
            "benchmark": "corpus",
            "mode": "single_query",
            "review_cadence": self.review_cadence,
            "state_boundary": "manager-selected pending native step interval",
            "relation_timing": "select_once_after_state_body",
            "hint_policy": "empty_state_hint; repair_hint_only",
            "worker_budget_scope": "Worker code-action steps; Manager actions excluded",
            "review_terminal_pending": self.review_terminal_pending,
        }

    def lifecycle_prompt(self) -> str:
        return _SINGLE_QUERY_LIFECYCLE.replace(
            "{{REVIEW_CADENCE}}", str(self.review_cadence)
        ).strip()

    def hint_state_ids(
        self, provisional_relation_ids: tuple[str, ...]
    ) -> tuple[str, ...]:
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
            raise ValueError("single-query corpus opens a state only after pending steps exist")
        if header.relations:
            raise ValueError("single-query relations are selected after writing the state body")
        if header.issue.strip():
            raise ValueError(
                "single-query OPEN_STATE contains only id and query constraints; "
                "derive the issue from the selected interval in UPDATE_STATE"
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
            raise ValueError("write the selected interval before relations")
        if draft.header.relations:
            raise ValueError("single-query OPEN_STATE cannot contain provisional relations")
        if not draft.issue.strip():
            raise ValueError("single-query state issue must come from the selected interval")
        if finalization.mode is not RelationFinalizationMode.SELECT:
            raise ValueError("single-query relations must be finalized posthoc with SELECT")
