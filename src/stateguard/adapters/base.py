from __future__ import annotations

from typing import Any, Protocol

from stateguard.core.models import TaskSpec
from stateguard.runtime.workspace import Workspace


class TaskAdapter(Protocol):
    """Dataset I/O only; agent and StateGuard control remain benchmark-neutral.

    A LongDS adapter returns one TaskSpec per turn from ``load_task_units`` and
    creates one workspace for the whole raw task. A single-query adapter returns
    exactly one unit. The caller reuses one harness within those units and creates
    a new harness/workspace for the next raw benchmark task.
    """

    def load_task_units(self, raw_task: Any) -> tuple[TaskSpec, ...]: ...

    def create_workspace(self, raw_task: Any) -> Workspace: ...

    def format_submission(self, raw_task: Any, final_answers: tuple[str, ...]) -> Any: ...
