from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stateguard.runtime.workspace import safe_clone


@dataclass(frozen=True)
class DABstepWorkspaceSnapshot:
    executor_state: Any
    executor_custom_tools: Any
    cleanup_events: tuple[dict[str, Any], ...]


class DABstepWorkspace:
    """View and checkpoint the official persistent Python interpreter."""

    def __init__(self, context_dir: Path) -> None:
        self.root = context_dir.expanduser().resolve(strict=True)
        self._worker: Any | None = None
        self.cleanup_events: list[dict[str, Any]] = []
        self._files = tuple(sorted(path for path in self.root.iterdir() if path.is_file()))

    def bind_worker(self, worker: Any) -> None:
        self._worker = worker

    def snapshot(self) -> DABstepWorkspaceSnapshot:
        executor = self._executor()
        return DABstepWorkspaceSnapshot(
            safe_clone(executor.state),
            safe_clone(executor.custom_tools),
            tuple(safe_clone(self.cleanup_events)),
        )

    def restore(self, snapshot: DABstepWorkspaceSnapshot) -> None:
        executor = self._executor()
        executor.state.clear()
        executor.state.update(safe_clone(snapshot.executor_state))
        executor.custom_tools.clear()
        executor.custom_tools.update(safe_clone(snapshot.executor_custom_tools))
        self.cleanup_events = list(safe_clone(snapshot.cleanup_events))

    def manifest(self) -> dict[str, Any]:
        variables: dict[str, str] = {}
        if self._worker is not None:
            for name, value in self._executor().state.items():
                if name != "print_outputs":
                    variables[str(name)] = type(value).__name__
        return {
            "runtime": "DABstep official smolagents LocalPythonInterpreter",
            "context_root": str(self.root),
            "data_files": {
                path.name: {"size": path.stat().st_size, "path": str(path)}
                for path in self._files
            },
            "python_variables": variables,
            "cleanup_events": list(self.cleanup_events[-3:]),
        }

    def stage_data_files(self, paths: tuple[str, ...]) -> dict[str, str]:
        return {Path(path).name: str(Path(path).resolve(strict=True)) for path in paths}

    def remove_variables(self, variables: tuple[str, ...]) -> None:
        state = self._executor().state
        removed: list[str] = []
        ignored: list[str] = []
        for versioned_name in variables:
            name = str(versioned_name).split("@", 1)[0].strip()
            if name and name in state and name != "print_outputs":
                state.pop(name, None)
                removed.append(name)
            else:
                ignored.append(str(versioned_name))
        self.cleanup_events.append(
            {"requested": list(variables), "removed_variables": removed, "ignored": ignored}
        )

    def _executor(self) -> Any:
        if self._worker is None:
            raise RuntimeError("DABstep workspace is not bound to its Worker")
        return self._worker.native_agent.python_executor
