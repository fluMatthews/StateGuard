from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from stateguard.agents.base import Agent
from stateguard.state.draft import DraftStore
from stateguard.state.graph import StateRelationGraph
from stateguard.state.store import StateStore
from stateguard.telemetry.artifacts import RunArtifactWriter

from .checkpoints import CheckpointManager
from .evidence_tools import build_manager_evidence_tools
from .executors import CodeExecutor, IsolatedProbeExecutor, TrustedPythonExecutor
from .trace import TraceBuffer
from .workspace import InMemoryWorkspace, Workspace


@dataclass
class StateGuardRuntime:
    """One identity-bound set of components for a benchmark task runtime."""

    id: str
    worker: Agent
    manager: Any | None
    workspace: Workspace
    state_store: StateStore
    graph: StateRelationGraph
    draft_store: DraftStore
    trace_buffer: TraceBuffer
    artifacts: RunArtifactWriter
    checkpoints: CheckpointManager
    worker_executor: CodeExecutor | None
    manager_probe_executor: IsolatedProbeExecutor | None

    @classmethod
    def create(
        cls,
        *,
        worker: Agent,
        manager: Any | None,
        workspace: Workspace,
        state_store: StateStore | None = None,
        graph: StateRelationGraph | None = None,
        draft_store: DraftStore | None = None,
        trace_buffer: TraceBuffer | None = None,
        artifacts: RunArtifactWriter | None = None,
        checkpoints: CheckpointManager | None = None,
        worker_executor: CodeExecutor | None = None,
        manager_probe_executor: IsolatedProbeExecutor | None = None,
    ) -> "StateGuardRuntime":
        store = state_store if state_store is not None else StateStore()
        relation_graph = graph if graph is not None else StateRelationGraph()
        drafts = draft_store if draft_store is not None else DraftStore()
        trace = trace_buffer if trace_buffer is not None else TraceBuffer()
        writer = artifacts if artifacts is not None else RunArtifactWriter()
        components: dict[str, Any] = {
            "worker": worker,
            "workspace": workspace,
            "state_store": store,
            "state_graph": relation_graph,
            "state_draft": drafts,
            "trace_buffer": trace,
        }
        checkpoint_manager = (
            checkpoints if checkpoints is not None else CheckpointManager(components)
        )
        _assert_checkpoint_binding(checkpoint_manager, components)
        existing_worker_executor = getattr(worker, "worker_executor", None)
        if worker_executor is None:
            worker_executor = existing_worker_executor
        elif (
            existing_worker_executor is not None
            and existing_worker_executor is not worker_executor
        ):
            raise ValueError("worker is already bound to a different executor")
        if isinstance(workspace, InMemoryWorkspace):
            worker_executor = worker_executor or TrustedPythonExecutor(workspace)
            manager_probe_executor = (
                manager_probe_executor or IsolatedProbeExecutor(workspace)
            )
        _assert_executor_binding(worker_executor, workspace, "worker_executor")
        bind_executor = getattr(worker, "bind_executor", None)
        if worker_executor is not None and callable(bind_executor):
            bind_executor(worker_executor)
        if manager_probe_executor is not None:
            _assert_executor_binding(
                manager_probe_executor,
                workspace,
                "manager_probe_executor",
                workspace_attribute="worker_workspace",
            )

        # A StateManagerAgent with no explicitly supplied tools receives the
        # evidence tools bound to this exact store/workspace runtime.
        tools = getattr(manager, "tools", None)
        if manager is not None and tools is not None:
            bound_tools = build_manager_evidence_tools(
                state_store=store,
                workspace=workspace,
                probe_executor=manager_probe_executor,
            )
            if len(tools) == 0:
                manager.tools = bound_tools
            elif (
                tools.bindings.get("state_store") is store
                and tools.bindings.get("workspace") is workspace
            ):
                pass
            else:
                reserved = tools.names().intersection(bound_tools.names())
                if reserved:
                    raise ValueError(
                        "manager evidence tools are bound to an unknown/different runtime: "
                        f"{sorted(reserved)}"
                    )
                for tool in bound_tools.tools():
                    tools.register(tool)
                tools.bindings.update(bound_tools.bindings)

        return cls(
            id=uuid4().hex,
            worker=worker,
            manager=manager,
            workspace=workspace,
            state_store=store,
            graph=relation_graph,
            draft_store=drafts,
            trace_buffer=trace,
            artifacts=writer,
            checkpoints=checkpoint_manager,
            worker_executor=worker_executor,
            manager_probe_executor=manager_probe_executor,
        )

    def assert_consistent(self) -> None:
        _assert_checkpoint_binding(
            self.checkpoints,
            {
                "worker": self.worker,
                "workspace": self.workspace,
                "state_store": self.state_store,
                "state_graph": self.graph,
                "state_draft": self.draft_store,
                "trace_buffer": self.trace_buffer,
            },
        )
        _assert_executor_binding(self.worker_executor, self.workspace, "worker_executor")
        worker_bound_executor = getattr(self.worker, "worker_executor", None)
        if (
            worker_bound_executor is not None
            and worker_bound_executor is not self.worker_executor
        ):
            raise ValueError("worker and runtime reference different executors")
        if self.manager_probe_executor is not None:
            _assert_executor_binding(
                self.manager_probe_executor,
                self.workspace,
                "manager_probe_executor",
                workspace_attribute="worker_workspace",
            )


def _assert_checkpoint_binding(
    checkpoints: CheckpointManager,
    expected: dict[str, Any],
) -> None:
    missing = set(expected).difference(checkpoints.components)
    if missing:
        raise ValueError(f"checkpoint manager is missing runtime components: {sorted(missing)}")
    mismatched = [
        name
        for name, component in expected.items()
        if checkpoints.components[name] is not component
    ]
    if mismatched:
        raise ValueError(
            "checkpoint manager is bound to different task-runtime objects: "
            f"{sorted(mismatched)}"
        )


def _assert_executor_binding(
    executor: Any | None,
    workspace: Workspace,
    label: str,
    *,
    workspace_attribute: str = "workspace",
) -> None:
    if executor is None:
        return
    bound_workspace = getattr(executor, workspace_attribute, workspace)
    if bound_workspace is not workspace:
        raise ValueError(f"{label} is bound to a different task workspace")
