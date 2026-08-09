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

from .dataset import LongDSPublicTurn
from .prompts import render_turn_messages


@dataclass
class LongDSWorkflow:
    """LongDS query-first, one-state-per-turn lifecycle."""

    turns: dict[str, LongDSPublicTurn]
    system_prompt_template: str

    def start(self, task: TaskSpec) -> None:
        self._turn(task)

    def review_before_worker(self) -> bool:
        # LongDS must choose provisional relations before this turn runs.
        return True

    def prepare_worker(
        self, worker: Agent, task: TaskSpec, workspace: Workspace
    ) -> None:
        del workspace
        turn = self._turn(task)
        messages = render_turn_messages(
            system_prompt_template=self.system_prompt_template,
            data_root=turn.data_root,
            context=turn.context,
            question=turn.question,
            first_turn=not bool(worker.messages),
        )
        if worker.messages:
            worker.continue_turn(messages)
        else:
            worker.start(messages)

    def should_review(self, step: ReActStep, steps_since_review: int) -> bool:
        del steps_since_review
        return step.done

    def review_event_type(self, step: ReActStep) -> str:
        del step
        return "TURN_END"

    def hint_state_ids(self, provisional_relation_ids: tuple[str, ...]) -> tuple[str, ...]:
        return provisional_relation_ids

    def lifecycle_prompt(self) -> str:
        return (
            files("stateguard.adapters.longds")
            .joinpath("manager_lifecycle.txt")
            .read_text(encoding="utf-8")
            .strip()
        )

    def manager_context(self) -> dict[str, Any]:
        return {
            "benchmark": "longds",
            "mode": "turn",
            "state_boundary": "one complete LongDS turn",
            "relation_timing": "query_first_then_confirm_or_reselect",
            "hint_policy": "provisional_relation_states",
            "pause_policy": "after official Worker turn completion only",
            "worker_budget_scope": "per_turn; Manager actions do not consume it",
        }

    def validate_state_open(
        self, header: StateHeader, untraced_steps: tuple[ReActStep, ...]
    ) -> None:
        if untraced_steps:
            raise ValueError("LongDS requires OPEN_STATE before the current turn trace")
        if not header.relations:
            raise ValueError("LongDS requires query-first provisional relations")
        if not header.issue.strip():
            raise ValueError("LongDS state header requires the query-defined issue")

    def validate_relation_finalization(
        self,
        *,
        draft: StateDraft,
        finalization: RelationFinalization,
        untraced_steps: tuple[ReActStep, ...],
    ) -> None:
        if not draft.traced_step_ids or untraced_steps:
            raise ValueError("write the complete checked turn before finalizing LongDS relations")
        if finalization.mode is RelationFinalizationMode.CONFIRM:
            if set(finalization.relations) != set(draft.header.relations):
                raise ValueError("CONFIRM must preserve provisional LongDS relations")
            return
        if finalization.mode is RelationFinalizationMode.RESELECT:
            if set(finalization.relations) == set(draft.header.relations):
                raise ValueError("RESELECT must replace conflicting provisional relations")
            return
        raise ValueError("LongDS relation finalization must be CONFIRM or RESELECT")

    def _turn(self, task: TaskSpec) -> LongDSPublicTurn:
        try:
            return self.turns[task.id]
        except KeyError as exc:
            raise KeyError(f"unknown LongDS unit: {task.id}") from exc
