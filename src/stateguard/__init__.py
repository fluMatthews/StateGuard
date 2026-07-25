"""StateGuard public API."""

from .adapters.flow import FixedStepFlowAdapter, TurnFlowAdapter
from .adapters.runner import TaskRunResult, run_task_units
from .agents.manager import StateManagerAgent
from .agents.worker import WorkerAgent
from .core.models import TaskSpec
from .harness.engine import StateGuardConfig, StateGuardHarness, StateGuardResult
from .runtime.bundle import StateGuardRuntime
from .state.models import AnalyticalState, StateRelationType, VariableRef
from .state.draft import (
    RelationFinalization,
    RelationFinalizationMode,
    StateHeader,
    StateUpdate,
)
from .state.store import StateStore
from .validation.models import AnalyticalEvidence, ErrorHint, ManagerAction, ManagerCommand, ManagerDecision

__all__ = [
    "AnalyticalState",
    "AnalyticalEvidence",
    "ErrorHint",
    "FixedStepFlowAdapter",
    "ManagerAction",
    "ManagerCommand",
    "ManagerDecision",
    "RelationFinalization",
    "RelationFinalizationMode",
    "StateGuardConfig",
    "StateGuardHarness",
    "StateGuardResult",
    "StateGuardRuntime",
    "StateHeader",
    "StateManagerAgent",
    "StateRelationType",
    "StateStore",
    "StateUpdate",
    "TaskSpec",
    "TaskRunResult",
    "TurnFlowAdapter",
    "VariableRef",
    "WorkerAgent",
    "run_task_units",
]
