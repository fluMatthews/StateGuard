from __future__ import annotations

from collections.abc import Iterable
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

    def capture(
        self,
        label: str,
        *,
        components: Iterable[str] | None = None,
    ) -> CheckpointRef:
        selected = tuple(self.components) if components is None else tuple(components)
        unknown = set(selected).difference(self.components)
        if unknown:
            raise KeyError(f"unknown checkpoint components: {sorted(unknown)}")
        if len(set(selected)) != len(selected):
            raise ValueError("checkpoint components must be unique")
        checkpoint_id = uuid4().hex
        self._snapshots[checkpoint_id] = {
            name: safe_clone(self.components[name].snapshot())
            for name in selected
        }
        self._labels[label] = checkpoint_id
        return CheckpointRef(checkpoint_id, label)

    def restore(self, reference: CheckpointRef | str) -> None:
        checkpoint_id = self._resolve(reference)
        snapshots = self._snapshots[checkpoint_id]
        restored: list[tuple[Snapshotable, Any]] = []
        try:
            for name, snapshot in snapshots.items():
                component = self.components[name]
                before = component.snapshot()
                component.restore(safe_clone(snapshot))
                restored.append((component, before))
        except Exception:
            for component, before in reversed(restored):
                component.restore(before)
            raise

    def by_label(self, label: str) -> CheckpointRef | None:
        checkpoint_id = self._labels.get(label)
        return CheckpointRef(checkpoint_id, label) if checkpoint_id else None

    @property
    def retained_count(self) -> int:
        return len(self._snapshots)

    def is_retained(self, reference: CheckpointRef | str) -> bool:
        value = reference.id if isinstance(reference, CheckpointRef) else reference
        return value in self._snapshots or value in self._labels

    def release(self, reference: CheckpointRef | str) -> None:
        """Drop a short-lived transaction snapshot after commit or rollback."""
        checkpoint_id = self._resolve(reference)
        self._snapshots.pop(checkpoint_id, None)
        self._labels = {
            label: stored_id
            for label, stored_id in self._labels.items()
            if stored_id != checkpoint_id
        }

    def _resolve(self, reference: CheckpointRef | str) -> str:
        value = reference.id if isinstance(reference, CheckpointRef) else reference
        if value in self._snapshots:
            return value
        if value in self._labels:
            return self._labels[value]
        raise KeyError(f"unknown checkpoint: {value}")
