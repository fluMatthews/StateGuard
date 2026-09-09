from __future__ import annotations

from typing import Any, Protocol

from stateguard.agents.base import Agent
from stateguard.core.events import ReActStep
from stateguard.core.models import TaskSpec
from stateguard.runtime.workspace import Workspace
from stateguard.state.draft import RelationFinalization, StateDraft, StateHeader


class BenchmarkWorkflow(Protocol):
    """Benchmark-owned lifecycle hooks used by the mechanism-only harness.

    A workflow owns native worker initialization, pause cadence, hint policy, and
    benchmark-specific manager instructions.  It does not own StateGuard state,
    repair, or commit semantics.
    """

    def start(self, task: TaskSpec) -> None: ...

    def review_before_worker(self) -> bool: ...

    def prepare_worker(
        self,
        worker: Agent,
        task: TaskSpec,
        workspace: Workspace,
    ) -> None: ...

    def should_review(self, step: ReActStep, steps_since_review: int) -> bool: ...

    def review_event_type(self, step: ReActStep) -> str: ...

    def manager_context(self) -> dict[str, Any]: ...

    def lifecycle_prompt(self) -> str: ...

    # Optional. A single-query flow has no state hint at OPEN_STATE, so the
    # Worker never learns what the Manager has already confirmed. Returning
    # True lets the harness hand it the newest committed states on resume.
    # Turn flows leave this unimplemented and are unaffected.
    def resumes_with_state_summary(self) -> bool: ...

    def validate_state_open(
        self,
        header: StateHeader,
        untraced_steps: tuple[ReActStep, ...],
    ) -> None: ...

    def hint_state_ids(
        self, provisional_relation_ids: tuple[str, ...]
    ) -> tuple[str, ...]: ...

    def validate_relation_finalization(
        self,
        *,
        draft: StateDraft,
        finalization: RelationFinalization,
        untraced_steps: tuple[ReActStep, ...],
    ) -> None: ...


class BenchmarkAdapter(Protocol):
    """Facade implemented once per benchmark package.

    Concrete task/result types remain benchmark-native; the shared contract is
    deliberately small so dataset schemas and official evaluators are not flattened.
    """

    def load_tasks(self, **selection: Any) -> tuple[Any, ...]: ...

    def run_task(self, task: Any, *, manager: Any | None = None, **kwargs: Any) -> Any: ...

    def judge(self, result: Any, **kwargs: Any) -> Any: ...
