from __future__ import annotations

from dataclasses import dataclass

from stateguard.core.models import TaskSpec
from stateguard.harness.engine import StateGuardHarness, StateGuardResult


@dataclass(frozen=True)
class TaskRunResult:
    """Results for one raw benchmark task (one or many task units)."""

    units: tuple[StateGuardResult, ...]

    @property
    def final_answers(self) -> tuple[str, ...]:
        return tuple(unit.final_answer for unit in self.units)


def run_task_units(
    harness: StateGuardHarness,
    units: tuple[TaskSpec, ...],
) -> TaskRunResult:
    """Run one benchmark task without imposing benchmark-specific semantics.

    Reusing the harness preserves the worker/workspace/store for LongDS turns.
    A single-query benchmark supplies one unit, so the same function degenerates
    to one ordinary ReAct run.
    """
    if not units:
        raise ValueError("a benchmark task must contain at least one TaskSpec unit")
    return TaskRunResult(tuple(harness.run(unit) for unit in units))
