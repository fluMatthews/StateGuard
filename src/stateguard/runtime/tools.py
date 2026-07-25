from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from stateguard.core.models import ToolResult


class Tool(Protocol):
    name: str
    description: str

    def invoke(self, arguments: dict[str, Any]) -> ToolResult: ...

    def schema(self) -> dict[str, Any]: ...


@dataclass
class FunctionTool:
    name: str
    description: str
    function: Callable[..., Any]
    parameters: dict[str, Any] | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters or {"type": "object", "additionalProperties": True},
        }

    def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            value = self.function(**arguments)
            if isinstance(value, ToolResult):
                return value
            data = value if isinstance(value, dict) else {}
            return ToolResult(self.name, True, str(value), data=data)
        except Exception as exc:  # tool errors are observations, not harness failures
            return ToolResult(self.name, False, "", error=f"{type(exc).__name__}: {exc}")


class ToolRegistry:
    def __init__(
        self,
        tools: list[Tool] | None = None,
        *,
        bindings: dict[str, Any] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self.bindings = dict(bindings or {})
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, False, "", error=f"unknown tool: {name}")
        return tool.invoke(arguments)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def clone(self) -> "ToolRegistry":
        """Give each agent an independent registry over the same tool objects."""
        return ToolRegistry(list(self._tools.values()), bindings=self.bindings)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
