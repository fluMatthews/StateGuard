from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from .workspace import safe_clone


class Snapshotable(Protocol):
    def snapshot(self) -> Any: ...

    def restore(self, snapshot: Any) -> None: ...


@dataclass(frozen=True)
class CheckpointRef:
    id: str
    label: str


class CheckpointManager:
    """Atomic composite checkpoints across conversation, workspace, store, and graph."""

    def __init__(self, components: dict[str, Snapshotable]) -> None:
        self.components = dict(components)
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._labels: dict[str, str] = {}

    def capture(self, label: str) -> CheckpointRef:
        checkpoint_id = uuid4().hex
        self._snapshots[checkpoint_id] = {
            name: safe_clone(component.snapshot())
            for name, component in self.components.items()
        }
        self._labels[label] = checkpoint_id
        return CheckpointRef(checkpoint_id, label)

    def restore(self, reference: CheckpointRef | str) -> None:
        checkpoint_id = self._resolve(reference)
        snapshots = self._snapshots[checkpoint_id]
        restored: list[tuple[Snapshotable, Any]] = []
        try:
            for name, component in self.components.items():
                before = component.snapshot()
                component.restore(safe_clone(snapshots[name]))
                restored.append((component, before))
        except Exception:
            for component, before in reversed(restored):
                component.restore(before)
            raise

    def by_label(self, label: str) -> CheckpointRef | None:
        checkpoint_id = self._labels.get(label)
        return CheckpointRef(checkpoint_id, label) if checkpoint_id else None

    def _resolve(self, reference: CheckpointRef | str) -> str:
        value = reference.id if isinstance(reference, CheckpointRef) else reference
        if value in self._snapshots:
            return value
        if value in self._labels:
            return self._labels[value]
        raise KeyError(f"unknown checkpoint: {value}")
