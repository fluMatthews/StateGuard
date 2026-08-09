from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .models import AgentAction, ToolResult


@dataclass(frozen=True)
class ReActStep:
    step_id: int
    action: AgentAction
    observation: ToolResult | None
    done: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_model_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def formed_explicit_result(self) -> bool:
        return bool(
            self.done
            or (self.observation and self.observation.data.get("important_result"))
        )


@dataclass(frozen=True)
class PendingInterval:
    start_step: int
    steps: tuple[ReActStep, ...]

    @property
    def end_step(self) -> int:
        return self.steps[-1].step_id if self.steps else self.start_step


@dataclass(frozen=True)
class ManagerFailure:
    phase: str
    event_type: str
    error_type: str
    message: str
    action_index: int | None = None
