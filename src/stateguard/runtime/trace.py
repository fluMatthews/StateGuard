from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum

from stateguard.core.events import ReActStep
from stateguard.core.models import to_jsonable


class TraceStatus(str, Enum):
    PENDING = "pending"
    DRAFTED = "drafted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PASSED = "passed"


@dataclass
class TraceRecord:
    step: ReActStep
    repair_attempt: int = 0
    status: TraceStatus = TraceStatus.PENDING

    def to_dict(self) -> dict:
        return {
            "step_id": self.step.step_id,
            "repair_attempt": self.repair_attempt,
            "status": self.status.value,
            "step": to_jsonable(self.step),
        }


class TraceBuffer:
    """Append-only branch ledger for worker steps.

    A state update tentatively assigns steps to the current draft.  They become
    authoritative only when that state commits.  A repair marks the current
    attempt's pending/drafted steps as rejected while retaining them for audit.
    """

    def __init__(self) -> None:
        self.records: list[TraceRecord] = []
        self.unit_id: str | None = None

    def start_unit(self, unit_id: str) -> None:
        unfinished = [
            record.step.step_id
            for record in self.records
            if record.status in {TraceStatus.PENDING, TraceStatus.DRAFTED}
        ]
        if unfinished:
            raise ValueError(
                f"cannot start unit {unit_id} with unfinished trace steps: {unfinished}"
            )
        self.unit_id = unit_id
        self.records = []

    @property
    def steps(self) -> list[ReActStep]:
        return [
            record.step
            for record in self.records
            if record.status is TraceStatus.PENDING
        ]

    def append(self, step: ReActStep, *, repair_attempt: int = 0) -> None:
        if any(record.step.step_id == step.step_id for record in self.records):
            raise ValueError(f"duplicate worker step id: {step.step_id}")
        self.records.append(TraceRecord(step, repair_attempt))

    def get(self, step_id: int) -> TraceRecord:
        """Read one exact worker step without exposing mutable trace state."""
        for record in self.records:
            if record.step.step_id == step_id:
                return copy.deepcopy(record)
        raise KeyError(f"unknown worker step: {step_id}")

    def validate_selection(
        self,
        step_ids: tuple[int, ...],
        *,
        allow_sparse: bool = False,
    ) -> tuple[int, ...]:
        """Validate a Manager trace selection and return deliberately omitted IDs.

        A normal state closes a contiguous prefix of the pending interval.  Once
        the current state has entered repair, the Manager may instead select an
        ordered subset of the rewritten interval: the largest selected step
        closes that interval prefix and unselected steps inside it are excluded
        from the state.  Pending steps after that boundary remain available for
        a later state.
        """
        pending_in_order = [
            record.step.step_id
            for record in self.records
            if record.status is TraceStatus.PENDING
        ]
        selected_list = list(step_ids)
        selected = set(selected_list)
        unknown = selected.difference(pending_in_order)
        if unknown:
            raise ValueError(f"manager traced unavailable worker steps: {sorted(unknown)}")
        if len(selected) != len(selected_list):
            raise ValueError("manager traced duplicate worker steps")

        if not allow_sparse:
            expected_prefix = pending_in_order[: len(selected_list)]
            if selected_list != expected_prefix:
                raise ValueError(
                    "a state interval must be a contiguous prefix beginning at the "
                    f"candidate start; expected {expected_prefix}, got {selected_list}"
                )
            return ()

        if not selected_list:
            return ()
        positions = {step_id: index for index, step_id in enumerate(pending_in_order)}
        if selected_list != sorted(selected_list, key=positions.__getitem__):
            raise ValueError(
                "a repaired state must trace worker steps in execution order; "
                f"got {selected_list}"
            )
        closed_prefix = pending_in_order[: positions[selected_list[-1]] + 1]
        return tuple(step_id for step_id in closed_prefix if step_id not in selected)

    def consume(
        self,
        step_ids: tuple[int, ...],
        *,
        allow_sparse: bool = False,
    ) -> None:
        omitted = set(
            self.validate_selection(step_ids, allow_sparse=allow_sparse)
        )
        selected = set(step_ids)
        for record in self.records:
            if record.step.step_id in selected:
                record.status = TraceStatus.DRAFTED
            elif record.step.step_id in omitted:
                record.status = TraceStatus.PASSED

    def accept(self, step_ids: tuple[int, ...]) -> None:
        selected = set(step_ids)
        drafted = {
            record.step.step_id
            for record in self.records
            if record.status is TraceStatus.DRAFTED
        }
        unknown = selected.difference(drafted)
        if unknown:
            raise ValueError(f"state commits non-drafted worker steps: {sorted(unknown)}")
        for record in self.records:
            if record.step.step_id in selected:
                record.status = TraceStatus.ACCEPTED

    def reject_attempt(self, repair_attempt: int) -> tuple[int, ...]:
        rejected: list[int] = []
        for record in self.records:
            if (
                record.repair_attempt == repair_attempt
                and record.status in {TraceStatus.PENDING, TraceStatus.DRAFTED}
            ):
                record.status = TraceStatus.REJECTED
                rejected.append(record.step.step_id)
        return tuple(rejected)

    def pass_uncommitted(self) -> tuple[int, ...]:
        passed: list[int] = []
        for record in self.records:
            if record.status in {TraceStatus.PENDING, TraceStatus.DRAFTED}:
                record.status = TraceStatus.PASSED
                passed.append(record.step.step_id)
        return tuple(passed)

    def history(self) -> tuple[dict, ...]:
        return tuple(record.to_dict() for record in self.records)

    def snapshot(self) -> tuple[str | None, list[TraceRecord]]:
        return copy.deepcopy((self.unit_id, self.records))

    def restore(self, snapshot: tuple[str | None, list[TraceRecord]]) -> None:
        self.unit_id, self.records = copy.deepcopy(snapshot)
