from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import AnalyticalState


class StateStore:
    """Authoritative verified-state store.

    A committed state is locked: commit never overwrites an ID, reads return
    defensive copies, and manager-facing tools expose no mutation operation.
    Snapshot/restore exists only for harness-owned transactional rollback.
    """

    def __init__(
        self,
        artifact_path: Path | None = None,
        state_dir: Path | None = None,
        index_path: Path | None = None,
    ) -> None:
        self._states: dict[str, AnalyticalState] = {}
        self._order: list[str] = []
        self.artifact_path = artifact_path
        self.state_dir = (
            state_dir
            if state_dir is not None
            else (artifact_path.parent / "states" if artifact_path is not None else None)
        )
        self.index_path = (
            index_path
            if index_path is not None
            else (
                artifact_path.parent / "state_index.json"
                if artifact_path is not None
                else None
            )
        )
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

    def relation_ids(self, state_id: str) -> tuple[str, ...]:
        """Return only the state's direct, one-hop upstream relation IDs."""
        return tuple(
            relation.related_state_id
            for relation in self.get(state_id).relations
            if relation.related_state_id is not None
        )

    def load_store_json(self) -> list[dict[str, Any]]:
        """Load the full aggregate committed-state JSON list for audit."""
        if self.artifact_path is None:
            return [state.to_dict() for state in self.all()]
        payload = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("store.json must contain a JSON list of states")
        return payload

    def load_state_index_json(self) -> list[dict[str, Any]]:
        """Load the compact relation-selection index supplied by the harness."""
        if self.index_path is None:
            return [_state_index_entry(state) for state in self.all()]
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("state_index.json must contain a JSON list")
        return payload

    def load_state_json(self, state_id: str) -> dict[str, Any]:
        """Load one committed state's dedicated JSON artifact by exact ID."""
        # Validate membership before deriving a path and keep reads limited to
        # committed states owned by this store.
        self.get(state_id)
        if self.state_dir is None:
            return self.get(state_id).to_dict()
        payload = json.loads((self.state_dir / f"{state_id}.json").read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("id") != state_id:
            raise ValueError(f"invalid dedicated state artifact for {state_id}")
        return payload

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
        states = [state.to_dict() for state in self.all()]
        if self.state_dir is None:
            raise RuntimeError("persistent StateStore requires a dedicated state directory")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        desired_files: set[Path] = set()
        for state in states:
            path = self.state_dir / f"{state['id']}.json"
            desired_files.add(path)
            _atomic_json(path, state)
        # Snapshot restore may remove a previously committed state. Keep the
        # aggregate list and dedicated files transactionally consistent.
        for path in self.state_dir.glob("S*.json"):
            if path not in desired_files:
                path.unlink()
        _atomic_json(self.artifact_path, states)
        if self.index_path is None:
            raise RuntimeError("persistent StateStore requires a state index path")
        _atomic_json(
            self.index_path,
            [_state_index_entry(state) for state in self.all()],
        )



def _state_index_entry(state: AnalyticalState) -> dict[str, Any]:
    return {
        "id": state.id,
        "issue": state.issue,
        "conclusions": [item.claim for item in state.conclusions],
    }


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
