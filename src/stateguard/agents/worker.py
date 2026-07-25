from __future__ import annotations

from stateguard.providers.base import ModelClient
from stateguard.runtime.executors import CodeExecutor, python_function_tool
from stateguard.runtime.tools import ToolRegistry

from .react import DEFAULT_REACT_SYSTEM_PROMPT, ReActAgent


class WorkerAgent(ReActAgent):
    """Standard ReAct worker; StateGuard behavior lives outside this class."""

    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry | None = None,
        *,
        executor: CodeExecutor | None = None,
        system_prompt: str = DEFAULT_REACT_SYSTEM_PROMPT,
        max_steps: int = 40,
    ) -> None:
        super().__init__(model, tools, system_prompt=system_prompt, max_steps=max_steps)
        self.worker_executor: CodeExecutor | None = None
        if executor is not None:
            self.bind_executor(executor)

    def bind_executor(self, executor: CodeExecutor) -> None:
        """Register the standard persistent ``python`` worker tool once."""
        if self.worker_executor is executor:
            return
        if self.worker_executor is not None or "python" in self.tools:
            raise ValueError("worker python tool is already bound to another executor")
        self.tools.register(python_function_tool(executor))
        self.tools.bindings["worker_executor"] = executor
        self.worker_executor = executor
