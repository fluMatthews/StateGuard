from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def to_jsonable(value: Any) -> Any:
    """Convert domain objects into JSON-compatible values."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")


@dataclass(frozen=True)
class TaskSpec:
    id: str
    query: str
    context: str = ""
    guidelines: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    data_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.query.strip():
            raise ValueError("task id and query must be non-empty")
        if any(not str(path).strip() for path in self.data_files):
            raise ValueError("task data file paths must be non-empty")

    def initial_prompt(self) -> str:
        parts = [self.query.strip()]
        if self.context.strip():
            parts.append(f"Context:\n{self.context.strip()}")
        if self.guidelines:
            parts.append("Guidelines:\n" + "\n".join(f"- {x}" for x in self.guidelines))
        return "\n\n".join(parts)


@dataclass(frozen=True)
class AgentAction:
    kind: str
    reasoning: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"tool", "final"}:
            raise ValueError("action kind must be 'tool' or 'final'")
        if self.kind == "tool" and not self.tool_name:
            raise ValueError("tool actions require tool_name")
        if self.kind == "final" and self.answer is None:
            raise ValueError("final actions require answer")


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    ok: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class AgentSession:
    messages: list[Message] = field(default_factory=list)
    step_count: int = 0
    done: bool = False
    final_answer: str | None = None

    def snapshot(self) -> "AgentSession":
        return copy.deepcopy(self)
