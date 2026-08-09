from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from stateguard.agents.base import Agent
from stateguard.runtime.checkpoints import CheckpointManager, CheckpointRef
from stateguard.runtime.workspace import Workspace
from stateguard.validation.models import ManagerAction, ManagerDecision


class RepairDirective(str, Enum):
    RETRY = "RETRY"
    RESTORED_ORIGINAL = "RESTORED_ORIGINAL"


@dataclass
class RepairSession:
    interval_start: CheckpointRef
    state_id: str | None = None
    original_branch: CheckpointRef | None = None
    attempts: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)

    def begin_state(self, state_id: str, interval_start: CheckpointRef) -> None:
        """Start an independent two-light/one-heavy budget for one state."""
        self.state_id = state_id
        self.interval_start = interval_start
        self.original_branch = None
        self.attempts = 0
        self.records.clear()

    def complete_state(self, clean_checkpoint: CheckpointRef) -> None:
        self.state_id = None
        self.interval_start = clean_checkpoint
        self.original_branch = None
        self.attempts = 0
        self.records.clear()

    def snapshot(self) -> dict[str, Any]:
        """Capture the small harness-owned repair ledger for action transactions."""
        return copy.deepcopy(
            {
                "interval_start": self.interval_start,
                "state_id": self.state_id,
                "original_branch": self.original_branch,
                "attempts": self.attempts,
                "records": self.records,
            }
        )

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.interval_start = copy.deepcopy(snapshot["interval_start"])
        self.state_id = copy.deepcopy(snapshot["state_id"])
        self.original_branch = copy.deepcopy(snapshot["original_branch"])
        self.attempts = int(snapshot["attempts"])
        self.records = copy.deepcopy(snapshot["records"])


class RepairController:
    """Execute the fixed two-light/one-heavy repair schedule."""

    def __init__(self, max_repairs: int = 3, light_repair_attempts: int = 2) -> None:
        if max_repairs < 1 or not 0 <= light_repair_attempts < max_repairs:
            raise ValueError("repair schedule requires 0 <= light attempts < max repairs")
        self.max_repairs = max_repairs
        self.light_repair_attempts = light_repair_attempts

    def apply(
        self,
        *,
        decision: ManagerDecision,
        session: RepairSession,
        current_branch: CheckpointRef,
        worker: Agent,
        workspace: Workspace,
    ) -> RepairDirective:
        if decision.action is not ManagerAction.REPAIR:
            raise ValueError("repair controller received a non-repair decision")
        if session.state_id is None:
            raise ValueError("repair must be bound to one open analytical state")
        if session.original_branch is None:
            session.original_branch = current_branch
        if session.attempts >= self.max_repairs:
            raise ValueError("repair schedule exhausted; manager must ABANDON_STATE")

        session.attempts += 1
        # The manager decides whether the same evidenced error warrants another
        # repair. Once triggered, the method fixes the repair phase by attempt.
        use_heavy = session.attempts > self.light_repair_attempts
        # Repair attempts are append-only: do not rewrite the worker's existing
        # conversation or restore a checkpoint here.  The heavy attempt differs
        # only by deleting the manager-identified erroneous variables before the
        # same structured hint is appended.
        if use_heavy:
            remove_variables = (
                decision.cleanup.remove_variables
                or decision.error_hint.error_variable
            )
            workspace.remove_variables(remove_variables)
        else:
            remove_variables = ()
        worker.inject_observation(
            decision.error_hint.as_observation(),
            metadata={
                "stateguard": "heavy_repair" if use_heavy else "light_repair",
                "repair_attempt": session.attempts,
            },
        )
        session.records.append(
            {
                "attempt": session.attempts,
                "mode": "heavy" if use_heavy else "light",
                "context_policy": "append_only",
                "removed_variables": list(remove_variables),
                "decision": decision,
            }
        )
        return RepairDirective.RETRY

    @staticmethod
    def abandon_state(
        session: RepairSession,
        checkpoints: CheckpointManager,
    ) -> RepairDirective:
        """Restore the first erroneous branch after all scheduled retries fail."""
        if session.original_branch is None:
            raise ValueError("ABANDON_STATE requires an active repair chain")
        checkpoints.restore(session.original_branch)
        session.records.append({"result": "repair_exhausted_restore_original_and_abandon_state"})
        return RepairDirective.RESTORED_ORIGINAL

    @staticmethod
    def abstain(session: RepairSession, checkpoints: CheckpointManager) -> RepairDirective:
        if session.original_branch is not None:
            checkpoints.restore(session.original_branch)
            session.records.append({"result": "manager_abstain_restore_original"})
        else:
            checkpoints.restore(session.interval_start)
            session.records.append({"result": "manager_abstain_restore_interval_start"})
        return RepairDirective.RESTORED_ORIGINAL
