from __future__ import annotations

import copy
import keyword
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .executor import InstrumentedToolGroup, LongDSExecutionRecord


@dataclass(frozen=True)
class LongDSWorkspaceSnapshot:
    executions: tuple[LongDSExecutionRecord, ...]
    operations: tuple["LongDSNamespaceOperation", ...]
    cleanup_events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LongDSNamespaceOperation:
    kind: str
    code: str


class LongDSWorkspace:
    """StateGuard control plane over the official persistent DSGym kernel.

    DSGym has no namespace snapshot endpoint. Checkpoint restore therefore restarts
    the same allocated kernel and replays the exact submitted-code prefix. This is
    sufficient for deterministic analytical cells and is explicitly reported in the
    manifest; benchmark data is never copied or modified by the adapter.
    """

    checkpoint_mode = "kernel_restart_and_code_replay"

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.expanduser().resolve()
        self.executions: list[LongDSExecutionRecord] = []
        self.operations: list[LongDSNamespaceOperation] = []
        self._tool_group: InstrumentedToolGroup | None = None
        self._environment: Any | None = None
        self.cleanup_events: list[dict[str, Any]] = []

    @property
    def last_execution(self) -> LongDSExecutionRecord | None:
        return self.executions[-1] if self.executions else None

    def bind_environment(
        self,
        environment: Any,
        clean_output: Callable[[list[Any]], str],
    ) -> None:
        if self._environment is not None and self._environment is not environment:
            raise ValueError("LongDSWorkspace cannot bind two Worker environments")
        self._environment = environment
        if isinstance(environment.tool_group, InstrumentedToolGroup):
            self._tool_group = environment.tool_group
            return
        self._tool_group = InstrumentedToolGroup(
            environment.tool_group,
            clean_output,
            self._record_execution,
        )
        environment.tool_group = self._tool_group

    def _record_execution(self, execution: LongDSExecutionRecord) -> None:
        self.executions.append(execution)
        self.operations.append(LongDSNamespaceOperation("worker", execution.code))

    def snapshot(self) -> LongDSWorkspaceSnapshot:
        return LongDSWorkspaceSnapshot(
            tuple(copy.deepcopy(self.executions)),
            tuple(copy.deepcopy(self.operations)),
            tuple(copy.deepcopy(self.cleanup_events)),
        )

    def restore(self, snapshot: LongDSWorkspaceSnapshot) -> None:
        desired = list(copy.deepcopy(snapshot.executions))
        desired_operations = list(copy.deepcopy(snapshot.operations))
        desired_cleanup_events = list(copy.deepcopy(snapshot.cleanup_events))
        # Conversation-only and control-plane transactions may ask the composite
        # checkpoint manager to restore an unchanged workspace.  The submitted
        # operation journal is the namespace source of truth: when it is already
        # identical, restoring bookkeeping must not restart/replay the Worker kernel.
        if self.operations == desired_operations:
            self.executions = desired
            self.operations = desired_operations
            self.cleanup_events = desired_cleanup_events
            return
        if self._tool_group is None:
            if desired_operations:
                raise RuntimeError("cannot restore LongDS code before environment allocation")
            self.executions = []
            self.operations = []
            self.cleanup_events = desired_cleanup_events
            return
        self._restart_kernel()
        for operation in desired_operations:
            try:
                self._tool_group._execute(operation.code, record=False)
            except RuntimeError:
                # Failed cells can still have partial Python side effects, so replay them
                # and continue just as the original notebook did.
                pass
        self.executions = desired
        self.operations = desired_operations
        self.cleanup_events = desired_cleanup_events

    def manifest(self) -> dict[str, Any]:
        latest = self.last_execution
        return {
            "runtime": "DSGym AllocatedCodeEnv",
            "data_root": str(self.data_root),
            "worker_code_cells": len(self.executions),
            "namespace_operations": len(self.operations),
            "checkpoint_mode": self.checkpoint_mode,
            "latest_execution": (
                {
                    "attempted": latest.attempted,
                    "succeeded": latest.succeeded,
                    "error": latest.error,
                }
                if latest is not None
                else None
            ),
            "cleanup_events": copy.deepcopy(self.cleanup_events[-3:]),
        }

    def stage_data_files(self, paths: tuple[str, ...]) -> dict[str, str]:
        # LongDS exposes the official task data directory in SYSTEM_PROMPT. This
        # compatibility method deliberately does not copy benchmark data.
        return {Path(path).name: str(Path(path).expanduser().resolve()) for path in paths}

    def remove_variables(self, variables: tuple[str, ...]) -> None:
        if not variables:
            return
        if self._tool_group is None:
            raise RuntimeError("LongDS environment is not initialized")
        normalized: list[str] = []
        ignored: list[str] = []
        for raw_name in variables:
            name = re.sub(r"@S[1-9][0-9]*$", "", raw_name.strip())
            if name.isidentifier() and not keyword.iskeyword(name) and not name.startswith("__"):
                normalized.append(name)
            else:
                ignored.append(raw_name)
        normalized = list(dict.fromkeys(normalized))
        self.cleanup_events.append(
            {"requested": list(variables), "removed": normalized, "ignored": ignored}
        )
        if not normalized:
            return
        code = (
            "for __stateguard_name in "
            + repr(tuple(normalized))
            + ":\n    globals().pop(__stateguard_name, None)\n"
            "globals().pop('__stateguard_name', None)"
        )
        self._tool_group.execute_hidden(code)
        self.operations.append(LongDSNamespaceOperation("cleanup", code))

    def close(self) -> None:
        if self._environment is not None:
            self._environment.close()
            self._environment = None
            self._tool_group = None

    def _restart_kernel(self) -> None:
        assert self._tool_group is not None
        delegate = self._tool_group.delegate
        container_id = delegate.allocated_container
        if container_id is None:
            raise RuntimeError("LongDS container is not allocated")
        response = delegate.client.post(
            f"{delegate.manager_url}/session/{container_id}/restart"
        )
        response.raise_for_status()
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            ready = delegate.client.get(
                f"{delegate.manager_url}/session/{container_id}/ready"
            )
            if ready.is_success and ready.json().get("ready"):
                return
            time.sleep(0.5)
        raise TimeoutError("LongDS kernel did not become ready after restart")
