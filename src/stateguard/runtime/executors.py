from __future__ import annotations

import contextlib
import io
import traceback
from dataclasses import dataclass
from typing import Any, Protocol

from stateguard.core.models import ToolResult

from .tools import FunctionTool
from .workspace import InMemoryWorkspace, safe_clone


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    stdout: str
    stderr: str = ""
    error: str | None = None


class CodeExecutor(Protocol):
    def execute(self, code: str) -> ExecutionResult: ...


def python_function_tool(executor: CodeExecutor) -> FunctionTool:
    """Expose a persistent code executor through the standard ReAct tool API."""

    def execute_python(code: str) -> ToolResult:
        result = executor.execute(code)
        return ToolResult(
            tool_name="python",
            ok=result.ok,
            output=result.stdout,
            data={"stderr": result.stderr},
            error=result.error,
        )

    return FunctionTool(
        name="python",
        description=(
            "Execute Python code in the persistent worker workspace. Use print() "
            "to expose results. Real task files are available through the "
            "data_files mapping."
        ),
        function=execute_python,
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    )


class TrustedPythonExecutor:
    """In-process executor for trusted experiments/tests, never for untrusted code."""

    def __init__(self, workspace: InMemoryWorkspace) -> None:
        self.workspace = workspace

    def execute(self, code: str) -> ExecutionResult:
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            compiled = compile(code, "<stateguard-executor>", "exec")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compiled, self.workspace.variables, self.workspace.variables)
            return ExecutionResult(True, stdout.getvalue(), stderr.getvalue())
        except Exception as exc:
            return ExecutionResult(
                False,
                stdout.getvalue(),
                stderr.getvalue(),
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )


class IsolatedProbeExecutor:
    """Run manager probes against a disposable clone of worker variables."""

    def __init__(self, worker_workspace: InMemoryWorkspace) -> None:
        self.worker_workspace = worker_workspace

    def execute(self, code: str) -> ExecutionResult:
        clone = InMemoryWorkspace(
            variables=safe_clone(self.worker_workspace.variables),
            artifacts=safe_clone(self.worker_workspace.artifacts),
            data_files=safe_clone(self.worker_workspace.data_files),
        )
        return TrustedPythonExecutor(clone).execute(code)
