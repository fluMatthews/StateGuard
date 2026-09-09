from __future__ import annotations

import copy
from dataclasses import InitVar, dataclass, field, replace
from enum import Enum
from typing import Any

from stateguard.core.models import to_jsonable

from .models import (
    AnalyticalState,
    CURRENT_STATE_VERSION,
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
            # Same reason as RelationFinalization: __post_init__ already explains
            # what a missing id means, and a bare KeyError would waste the one
            # correction retry on an unactionable message.
            id=str(value.get("id", "$AUTO_STATE")),
            constraints=tuple(Constraint.from_dict(item) for item in value.get("constraints", [])),
            relations=tuple(StateRelation.from_dict(item) for item in value.get("relations", [])),
            issue=str(value.get("issue", "")),
        )


@dataclass(frozen=True)
class SourceInterval:
    """Inclusive Worker-step span observed when writing one analytical state."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < 1:
            raise ValueError("source interval step IDs must be positive")
        if self.start > self.end:
            raise ValueError("source interval start must not exceed end")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceInterval":
        return cls(start=int(value["start"]), end=int(value["end"]))

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class StateUpdate:
    issue: str | None = None
    used_variables: tuple[VariableRef, ...] = ()
    conclusions: tuple[Conclusion, ...] = ()
    source_interval: SourceInterval | None = None
    # Harness-owned binding. Manager JSON must use source_interval instead.
    traced_step_ids: tuple[int, ...] = field(default=(), repr=False)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "StateUpdate":
        value = value or {}
        if "traced_step_ids" in value:
            raise ValueError(
                "Manager must not enumerate traced_step_ids; use source_interval "
                "for single-query state boundaries"
            )
        return cls(
            issue=str(value["issue"]) if value.get("issue") is not None else None,
            used_variables=tuple(VariableRef.from_dict(item) for item in value.get("used_variables", [])),
            conclusions=tuple(Conclusion.from_value(item) for item in value.get("conclusions", [])),
            source_interval=(
                SourceInterval.from_dict(value["source_interval"])
                if value.get("source_interval") is not None
                else None
            ),
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
    reason: InitVar[str | None] = None
    conflict_evidence: tuple[str, ...] = ()

    def __post_init__(self, reason: str | None) -> None:
        del reason
        validate_relation_set(
            self.relations,
            allow_empty=self.mode is RelationFinalizationMode.CONFIRM,
        )
        if self.mode is RelationFinalizationMode.RESELECT and not self.conflict_evidence:
            raise ValueError("relation reselection requires explicit conflict evidence")
        if self.mode is not RelationFinalizationMode.RESELECT and self.conflict_evidence:
            raise ValueError("conflict evidence is only valid for relation reselection")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RelationFinalization":
        return cls(
            mode=RelationFinalizationMode(str(value["mode"]).lower()),
            relations=tuple(
                StateRelation.from_dict(item) for item in value.get("relations", ())
            ),
            conflict_evidence=tuple(str(item) for item in value.get("conflict_evidence", [])),
        )


@dataclass
class StateDraft:
    header: StateHeader
    issue: str = ""
    used_variables: dict[str, VariableRef] = field(default_factory=dict)
    conclusions: dict[str, Conclusion] = field(default_factory=dict)
    source_interval: SourceInterval | None = None
    traced_step_ids: list[int] = field(default_factory=list)
    final_relations: tuple[StateRelation, ...] | None = None
    relation_finalization_mode: RelationFinalizationMode | None = None
    relation_conflict_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.issue = self.header.issue

    def apply(self, update: StateUpdate) -> None:
        if update.issue is not None:
            if not update.issue.strip():
                raise ValueError("state issue must be non-empty when written")
            self.issue = update.issue
        for variable in update.used_variables:
            if variable.version == CURRENT_STATE_VERSION:
                variable = replace(variable, version=self.header.id)
            if variable.version != self.header.id:
                raise ValueError(
                    f"variable {variable.name} must use current state version "
                    f"{self.header.id}, got {variable.version}"
                )
            self.used_variables[variable.key] = variable
        for conclusion in update.conclusions:
            self.conclusions[conclusion.claim] = conclusion
        if update.source_interval is not None:
            if self.source_interval is None:
                self.source_interval = update.source_interval
            else:
                if update.source_interval.start != self.source_interval.start:
                    raise ValueError(
                        "a repaired or extended state must preserve its source interval "
                        f"start {self.source_interval.start}"
                    )
                if update.source_interval.end < self.source_interval.end:
                    raise ValueError("a state source interval may only extend forward")
                self.source_interval = update.source_interval
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
        self.relation_conflict_evidence = finalization.conflict_evidence

    def reset_content_for_retry(self) -> None:
        """Discard rejected attempt content while preserving query-first header."""
        self.issue = self.header.issue
        self.used_variables.clear()
        self.conclusions.clear()
        self.traced_step_ids.clear()
        self.final_relations = None
        self.relation_finalization_mode = None
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
            source_step_start=(
                self.source_interval.start if self.source_interval else None
            ),
            source_step_end=(
                self.source_interval.end if self.source_interval else None
            ),
            metadata=self._relation_metadata(),
        )

    def to_observation_dict(self) -> dict[str, Any]:
        """Compact live draft view; full draft metadata stays in artifacts."""
        return {
            "id": self.header.id,
            "issue": self.issue,
            "constraints": [item.to_dict() for item in self.header.constraints],
            "used_variables": [
                item.to_dict() for item in self.used_variables.values()
            ],
            "conclusions": [
                item.claim for item in self.conclusions.values()
            ],
            "relations": to_jsonable(self.relations),
            "source_interval": (
                self.source_interval.to_dict() if self.source_interval else None
            ),
            "relations_finalized": self.relations_finalized,
            "relation_finalization_mode": (
                self.relation_finalization_mode.value
                if self.relation_finalization_mode is not None
                else None
            ),
            "relation_conflict_evidence": list(
                self.relation_conflict_evidence
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        metadata = self._relation_metadata()
        return {
            "id": self.header.id,
            "issue": self.issue,
            "constraints": [item.to_dict() for item in self.header.constraints],
            "used_variables": [item.to_dict() for item in self.used_variables.values()],
            "conclusions": [item.claim for item in self.conclusions.values()],
            "relations": to_jsonable(self.relations),
            "source_step_start": (
                self.source_interval.start if self.source_interval else None
            ),
            "source_step_end": (
                self.source_interval.end if self.source_interval else None
            ),
            "status": "draft",
            "metadata": metadata,
            "lifecycle": "tracing",
            "source_interval": (
                self.source_interval.to_dict() if self.source_interval else None
            ),
            "bound_step_ids": list(self.traced_step_ids),
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
