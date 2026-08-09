from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from stateguard.runtime.executors import ExecutionResult


@dataclass(frozen=True)
class LongDSExecutionRecord:
    code: str
    attempted: bool
    succeeded: bool
    raw_outputs: tuple[Any, ...]
    cleaned_output: str
    error: str | None = None


class InstrumentedToolGroup:
    """Preserve DSGym-visible output while retaining structured execution facts."""

    def __init__(
        self,
        delegate: Any,
        clean_output: Callable[[list[Any]], str],
        on_execution: Callable[[LongDSExecutionRecord], None],
    ) -> None:
        self.delegate = delegate
        self.clean_output = clean_output
        self.on_execution = on_execution
        self.record_enabled = True

    @property
    def allocated_container(self) -> Any:
        return self.delegate.allocated_container

    def allocate_container(self) -> Any:
        return self.delegate.allocate_container()

    def deallocate_container(self) -> Any:
        return self.delegate.deallocate_container()

    def get_tool_names(self) -> list[str]:
        return self.delegate.get_tool_names()

    def execute_code(self, code: str) -> str:
        return self._execute(code, record=self.record_enabled)

    def execute_hidden(self, code: str) -> LongDSExecutionRecord:
        before = self.record_enabled
        self.record_enabled = False
        try:
            self._execute(code, record=False)
            return self.last_execution
        finally:
            self.record_enabled = before

    def _execute(self, code: str, *, record: bool) -> str:
        if self.delegate.allocated_container is None:
            raise RuntimeError("No container allocated")
        try:
            response = self.delegate.client.post(
                f"{self.delegate.manager_url}/session/"
                f"{self.delegate.allocated_container}/execute",
                json={"code": code},
            )
            response.raise_for_status()
            raw_outputs = list(response.json().get("outputs", []))
            cleaned = self.clean_output(raw_outputs)
            errors = [
                item
                for item in raw_outputs
                if isinstance(item, dict) and item.get("type") == "error"
            ]
            error = _render_error(errors[0]) if errors else None
            execution = LongDSExecutionRecord(
                code=code,
                attempted=True,
                succeeded=not errors,
                raw_outputs=tuple(raw_outputs),
                cleaned_output=cleaned,
                error=error,
            )
            self.last_execution = execution
            if record:
                self.on_execution(execution)
            return cleaned
        except Exception as exc:
            error_msg = f"Failed to execute code: {exc}"
            response = getattr(exc, "response", None)
            if response is not None:
                error_msg += f" (Status: {response.status_code})"
            execution = LongDSExecutionRecord(
                code=code,
                attempted=True,
                succeeded=False,
                raw_outputs=(),
                cleaned_output="",
                error=error_msg,
            )
            self.last_execution = execution
            if record:
                self.on_execution(execution)
            raise RuntimeError(error_msg) from exc

    last_execution = LongDSExecutionRecord("", False, False, (), "")


class LongDSProbeExecutor:
    """Manager-only short-lived scratch container with real task-data handles."""

    def __init__(
        self,
        worker_workspace: Any,
        environment_factory: Callable[[], Any],
        clean_output: Callable[[list[Any]], str],
    ) -> None:
        self.worker_workspace = worker_workspace
        self.environment_factory = environment_factory
        self.clean_output = clean_output
        self._environment: Any | None = None
        self._tool_group: InstrumentedToolGroup | None = None
        self._data_files: dict[str, str] | None = None

    def execute(self, code: str) -> ExecutionResult:
        try:
            # A probe is an independent focused check, not a clone of the Worker
            # notebook.  Start empty, expose only real task-data paths, execute the
            # Manager's current check, and never replay Worker analytical cells.
            self.close()
            self._environment = self.environment_factory()
            self._tool_group = InstrumentedToolGroup(
                self._environment.tool_group,
                self.clean_output,
                lambda execution: None,
            )
            self._environment.tool_group = self._tool_group
            self._environment.init([])
            assert self._tool_group is not None
            data_root = self.worker_workspace.data_root
            data_files = self._task_data_files()
            bootstrap = self._tool_group.execute_hidden(
                f"DATA_ROOT = {str(data_root)!r}\n"
                f"data_files = {data_files!r}"
            )
            if not bootstrap.succeeded:
                return ExecutionResult(False, "", error=bootstrap.error)
            output = self._tool_group.execute_code(code)
            execution = self._tool_group.last_execution
            return ExecutionResult(
                execution.succeeded,
                output,
                error=execution.error,
            )
        except Exception as exc:
            return ExecutionResult(False, "", error=f"{type(exc).__name__}: {exc}")
        finally:
            self.close()

    def _task_data_files(self) -> dict[str, str]:
        """Lazily cache the fixed benchmark input-path mapping for this task."""
        if self._data_files is None:
            data_root = self.worker_workspace.data_root
            self._data_files = {
                str(path.relative_to(data_root)): str(path)
                for path in data_root.rglob("*")
                if path.is_file()
            }
        return dict(self._data_files)

    def close(self) -> None:
        if self._environment is not None:
            self._environment.close()
            self._environment = None
            self._tool_group = None


def _render_error(value: dict[str, Any]) -> str:
    name = value.get("name") or "ExecutionError"
    message = value.get("value") or value.get("message") or str(value)
    return f"{name}: {message}"
