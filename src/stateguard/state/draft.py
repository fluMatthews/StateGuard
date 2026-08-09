from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from stateguard.core.models import to_jsonable

from .models import (
    AnalyticalState,
    Conclusion,
    Constraint,
    StateRelation,
    VariableRef,
    validate_relation_set,
)


@dataclass(frozen=True)
class StateHeader:
    """Manager-written state identity and provisional relations."""

    id: str
    constraints: tuple[Constraint, ...]
    relations: tuple[StateRelation, ...]
    issue: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("state header id must be non-empty")
        validate_relation_set(self.relations, allow_empty=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StateHeader":
        return cls(
            id=str(value["id"]),
            constraints=tuple(Constraint.from_dict(item) for item in value.get("constraints", [])),
            relations=tuple(StateRelation.from_dict(item) for item in value.get("relations", [])),
            issue=str(value.get("issue", "")),
        )


@dataclass(frozen=True)
class StateUpdate:
    issue: str | None = None
    used_variables: tuple[VariableRef, ...] = ()
    conclusions: tuple[Conclusion, ...] = ()
    traced_step_ids: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "StateUpdate":
        value = value or {}
        return cls(
            issue=str(value["issue"]) if value.get("issue") is not None else None,
            used_variables=tuple(VariableRef.from_dict(item) for item in value.get("used_variables", [])),
            conclusions=tuple(Conclusion.from_value(item) for item in value.get("conclusions", [])),
            traced_step_ids=tuple(int(item) for item in value.get("traced_step_ids", [])),
        )


class RelationFinalizationMode(str, Enum):
    """How the final relation set was produced."""

    CONFIRM = "confirm"
    RESELECT = "reselect"
    SELECT = "select"


@dataclass(frozen=True)
class RelationFinalization:
    """Manager confirmation or evidence-based revision before state commit."""

    mode: RelationFinalizationMode
    relations: tuple[StateRelation, ...]
    reason: str
    conflict_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_relation_set(self.relations)
        if not self.reason.strip():
            raise ValueError("relation finalization requires a reason")
        if self.mode is RelationFinalizationMode.RESELECT and not self.conflict_evidence:
            raise ValueError("relation reselection requires explicit conflict evidence")
        if self.mode is not RelationFinalizationMode.RESELECT and self.conflict_evidence:
            raise ValueError("conflict evidence is only valid for relation reselection")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationFinalization":
        return cls(
            mode=RelationFinalizationMode(str(value["mode"]).lower()),
            relations=tuple(StateRelation.from_dict(item) for item in value["relations"]),
            reason=str(value["reason"]),
            conflict_evidence=tuple(str(item) for item in value.get("conflict_evidence", [])),
        )


@dataclass
class StateDraft:
    header: StateHeader
    issue: str = ""
    used_variables: dict[str, VariableRef] = field(default_factory=dict)
    conclusions: dict[str, Conclusion] = field(default_factory=dict)
    traced_step_ids: list[int] = field(default_factory=list)
    final_relations: tuple[StateRelation, ...] | None = None
    relation_finalization_mode: RelationFinalizationMode | None = None
    relation_finalization_reason: str = ""
    relation_conflict_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.issue = self.header.issue

    def apply(self, update: StateUpdate) -> None:
        if update.issue is not None:
            if not update.issue.strip():
                raise ValueError("state issue must be non-empty when written")
            self.issue = update.issue
        for variable in update.used_variables:
            if variable.version != self.header.id:
                raise ValueError(
                    f"variable {variable.name} must use current state version "
                    f"{self.header.id}, got {variable.version}"
                )
            self.used_variables[variable.key] = variable
        for conclusion in update.conclusions:
            self.conclusions[conclusion.claim] = conclusion
        for step_id in update.traced_step_ids:
            if step_id not in self.traced_step_ids:
                self.traced_step_ids.append(step_id)

    @property
    def relations(self) -> tuple[StateRelation, ...]:
        return self.final_relations or self.header.relations

    @property
    def relations_finalized(self) -> bool:
        return self.final_relations is not None

    def finalize_relations(self, finalization: RelationFinalization) -> None:
        self.final_relations = finalization.relations
        self.relation_finalization_mode = finalization.mode
        self.relation_finalization_reason = finalization.reason
        self.relation_conflict_evidence = finalization.conflict_evidence

    def reset_content_for_retry(self) -> None:
        """Discard rejected attempt content while preserving query-first header."""
        self.issue = self.header.issue
        self.used_variables.clear()
        self.conclusions.clear()
        self.traced_step_ids.clear()
        self.final_relations = None
        self.relation_finalization_mode = None
        self.relation_finalization_reason = ""
        self.relation_conflict_evidence = ()

    def _relation_metadata(self) -> dict[str, Any]:
        return {
            "provisional_relations": [
                {"type": item.type.value, "related_state_id": item.related_state_id}
                for item in self.header.relations
            ],
            "final_relations": [
                {"type": item.type.value, "related_state_id": item.related_state_id}
                for item in (self.final_relations or ())
            ],
            "relations_finalized": self.relations_finalized,
            "relation_finalization_mode": (
                self.relation_finalization_mode.value
                if self.relation_finalization_mode is not None
                else None
            ),
            "relation_finalization_reason": self.relation_finalization_reason,
            "relation_conflict_evidence": list(self.relation_conflict_evidence),
        }

    def to_state(self) -> AnalyticalState:
        return AnalyticalState(
            id=self.header.id,
            issue=self.issue,
            constraints=self.header.constraints,
            used_variables=tuple(self.used_variables.values()),
            conclusions=tuple(self.conclusions.values()),
            relations=self.relations,
            source_step_start=min(self.traced_step_ids) if self.traced_step_ids else None,
            source_step_end=max(self.traced_step_ids) if self.traced_step_ids else None,
            metadata=self._relation_metadata(),
        )

    def to_dict(self) -> dict[str, Any]:
        metadata = self._relation_metadata()
        return {
            "id": self.header.id,
            "issue": self.issue,
            "constraints": [item.to_dict() for item in self.header.constraints],
            "used_variables": [item.to_dict() for item in self.used_variables.values()],
            "conclusions": [item.claim for item in self.conclusions.values()],
            "relations": to_jsonable(self.relations),
            "source_step_start": min(self.traced_step_ids) if self.traced_step_ids else None,
            "source_step_end": max(self.traced_step_ids) if self.traced_step_ids else None,
            "status": "draft",
            "metadata": metadata,
            "lifecycle": "tracing",
            "traced_step_ids": list(self.traced_step_ids),
            **metadata,
        }


class DraftStore:
    """Single authoritative draft controlled by manager actions."""

    def __init__(self) -> None:
        self.current: StateDraft | None = None

    def open(self, header: StateHeader) -> StateDraft:
        if self.current is not None:
            raise ValueError(f"state {self.current.header.id} is still open")
        self.current = StateDraft(header)
        return self.current

    def update(self, update: StateUpdate) -> StateDraft:
        if self.current is None:
            raise ValueError("manager must OPEN_STATE before UPDATE_STATE")
        self.current.apply(update)
        return self.current

    def finalize_relations(self, finalization: RelationFinalization) -> StateDraft:
        if self.current is None:
            raise ValueError("manager must OPEN_STATE before FINALIZE_RELATIONS")
        self.current.finalize_relations(finalization)
        return self.current

    def close(self) -> AnalyticalState:
        if self.current is None:
            raise ValueError("manager must OPEN_STATE before COMMIT_STATE")
        if not self.current.relations_finalized:
            raise ValueError("manager must FINALIZE_RELATIONS before COMMIT_STATE")
        state = self.current.to_state()
        self.current = None
        return state

    def reset_content_for_retry(self) -> None:
        if self.current is not None:
            self.current.reset_content_for_retry()

    def discard(self) -> None:
        self.current = None

    def snapshot(self) -> StateDraft | None:
        return copy.deepcopy(self.current)

    def restore(self, snapshot: StateDraft | None) -> None:
        self.current = copy.deepcopy(snapshot)
