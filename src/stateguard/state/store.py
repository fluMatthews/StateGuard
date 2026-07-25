from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .models import AnalyticalState


class StateStore:
    """Authoritative verified-state store.

    A committed state is locked: commit never overwrites an ID, reads return
    defensive copies, and manager-facing tools expose no mutation operation.
    Snapshot/restore exists only for harness-owned transactional rollback.
    """

    def __init__(self, artifact_path: Path | None = None) -> None:
        self._states: dict[str, AnalyticalState] = {}
        self._order: list[str] = []
        self.artifact_path = artifact_path
        self._persist()

    def commit(self, state: AnalyticalState) -> None:
        if state.id in self._states:
            raise ValueError(f"state already committed: {state.id}")
        if state.status != "committed":
            raise ValueError("only a checked committed state may enter the verified store")
        for relation in state.relations:
            if relation.related_state_id and relation.related_state_id not in self._states:
                raise ValueError(
                    f"state {state.id} references unknown state "
                    f"{relation.related_state_id}"
                )
        self._states[state.id] = copy.deepcopy(state)
        self._order.append(state.id)
        self._persist()

    def get(self, state_id: str) -> AnalyticalState:
        return copy.deepcopy(self._states[state_id])

    def all(self) -> tuple[AnalyticalState, ...]:
        return tuple(copy.deepcopy(self._states[state_id]) for state_id in self._order)

    def catalog(self) -> list[dict[str, Any]]:
        return [state.as_state_hint() for state in self.all()]

    def relation_catalog(self) -> list[dict[str, Any]]:
        """Manager-visible state contents for evidence-based relation selection."""
        return [state.as_relation_view() for state in self.all()]

    def index(self) -> list[dict[str, Any]]:
        """Compact manager-visible index; manager selects related IDs directly."""
        return [
            {
                "id": state.id,
                "issue": state.issue,
                "variable_keys": [item.key for item in state.used_variables],
                "conclusions": [item.claim for item in state.conclusions],
                "relations": [
                    {"type": relation.type.value, "related_state_id": relation.related_state_id}
                    for relation in state.relations
                ],
            }
            for state in self.all()
        ]

    def variable(self, key: str) -> list[tuple[str, Any]]:
        matches: list[tuple[str, Any]] = []
        for state in self.all():
            for variable in state.used_variables:
                if variable.key == key or variable.name == key:
                    matches.append((state.id, variable))
        return matches

    def relation_ids(self, state_id: str) -> tuple[str, ...]:
        """Return only the state's direct, one-hop upstream relation IDs."""
        return tuple(
            relation.related_state_id
            for relation in self.get(state_id).relations
            if relation.related_state_id is not None
        )

    def snapshot(self) -> tuple[dict[str, AnalyticalState], list[str]]:
        return copy.deepcopy((self._states, self._order))

    def restore(self, snapshot: tuple[dict[str, AnalyticalState], list[str]]) -> None:
        self._states, self._order = copy.deepcopy(snapshot)
        self._persist()

    def __len__(self) -> int:
        return len(self._order)

    def _persist(self) -> None:
        if self.artifact_path is None:
            return
        payload = {
            "schema_version": "stateguard-state-v2",
            "states": [state.to_dict() for state in self.all()],
        }
        _atomic_json(self.artifact_path, payload)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
