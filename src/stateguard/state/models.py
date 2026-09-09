from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Iterable

from stateguard.core.models import to_jsonable


CURRENT_STATE_VERSION = "$CURRENT_STATE"


class StateRelationType(str, Enum):
    INIT = "init"
    PROGRESS = "progress"
    BRANCH = "branch"
    INVALIDATE = "invalidate"
    COMBINE = "combine"


@dataclass(frozen=True)
class Constraint:
    text: str
    code: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("constraint text must be non-empty")

    @classmethod
    def from_dict(cls, value: Any) -> "Constraint":
        # A bare string is accepted as the constraint text, mirroring
        # Conclusion.from_value. Managers routinely write a plain list of
        # strings, and indexing one raised an unactionable TypeError.
        if not isinstance(value, dict):
            return cls(text=str(value))
        return cls(
            text=str(value["text"]),
            # ``executable_predicate`` is accepted only when reading an older
            # artifact. New manager actions use the deliberately small
            # natural-language-plus-optional-code schema.
            code=value.get("code", value.get("executable_predicate")),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"text": self.text}
        if self.code:
            value["code"] = self.code
        return value


# What a Worker cannot cheaply rediscover is worth sending back to it; what it
# can is noise. Across every DE state written so far the split is clean and
# mechanical: two thirds of the variables are inventories of table or file
# names -- staging_files_written, marts, *_model_list -- which the Worker
# produced itself and can list with one shell command. The remainder is what
# it paid many steps to learn, and would have to pay again: that a source
# table holds admin_id in the team_id column, that a conversation_id column
# carries timestamps, that an email filter cut 3,004 rows down to 7.
#
# An inventory is recognizable without knowing any of that. Split on commas
# and every piece is a bare identifier -- letters, digits, underscores, dots.
# A finding never is: "email (lower(trim(property_email)) not null)" and
# "admin_id = id (3004 rows; ...)" carry spaces, parentheses, equals signs.
# The test is on the rendered shape rather than the Python type because the
# same inventories arrive both ways, half as lists and half as comma-joined
# strings. Requiring every piece to be bare, not most, is what keeps
# raw_schema -- a sentence followed by a column list -- on the useful side.
#
# 500 characters clears the longest finding seen (466: a note that the source
# column order departs from the contract, with the mapping it rebuilt).
# Scalars and booleans always survive: nothing distinguishes a bare
# marts_written=true from job_application_duplication=true except meaning, and
# four characters is not worth the risk of dropping the warning.
_HINT_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.\-]+$")
_HINT_VALUE_CHARS = 500


def _is_inventory(value: Any) -> bool:
    """True when the value is only a list of names."""
    if value is None or isinstance(value, (bool, int, float)):
        return False
    pieces = value if isinstance(value, list) else str(value).split(",")
    cleaned = [str(piece).strip().strip("'\"") for piece in pieces]
    cleaned = [piece for piece in cleaned if piece]
    if len(cleaned) < 3:
        return False
    return all(_HINT_IDENTIFIER.match(piece) for piece in cleaned)


def _hint_variable(variable: "VariableRef") -> dict[str, Any] | None:
    if _is_inventory(variable.value):
        return None
    value = variable.value
    if not isinstance(value, (bool, int, float)) and value is not None:
        rendered = value if isinstance(value, str) else to_jsonable(value)
        text = rendered if isinstance(rendered, str) else str(rendered)
        if len(text) > _HINT_VALUE_CHARS:
            value = f"{text[:_HINT_VALUE_CHARS]}…"
        else:
            value = rendered
    return {"name": variable.name, "value": value}


@dataclass(frozen=True)
class VariableRef:
    name: str
    version: str
    value: Any = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("variable requires a non-empty name")
        if not isinstance(self.version, str) or (
            self.version != CURRENT_STATE_VERSION
            and not re.fullmatch(r"S[1-9][0-9]*", self.version)
        ):
            raise ValueError("variable version must be the producing state ID, e.g. S3")

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VariableRef":
        # The Manager writes content; the harness owns the current draft ID and
        # attaches that version mechanically when this update is applied.
        raw_version = str(value.get("version", CURRENT_STATE_VERSION)).strip()
        return cls(
            name=str(value["name"]),
            version=raw_version,
            value=value.get("value"),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"name": self.name, "version": self.version}
        if self.value is not None:
            value["value"] = self.value
        return value


@dataclass(frozen=True)
class Conclusion:
    claim: str

    def __post_init__(self) -> None:
        if not self.claim.strip():
            raise ValueError("conclusion claim must be non-empty")

    @classmethod
    def from_value(cls, value: Any) -> "Conclusion":
        if isinstance(value, dict):  # backward-compatible artifact reader
            value = value["claim"]
        return cls(claim=str(value))


@dataclass(frozen=True)
class StateRelation:
    type: StateRelationType
    related_state_id: str | None = None

    def __post_init__(self) -> None:
        if self.type is StateRelationType.INIT and self.related_state_id is not None:
            raise ValueError("init relation cannot reference an earlier state")
        if self.type is not StateRelationType.INIT and not self.related_state_id:
            raise ValueError(f"{self.type.value} relation requires related_state_id")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StateRelation":
        return cls(
            type=StateRelationType(str(value["type"]).lower()),
            related_state_id=value.get("related_state_id"),
        )


def validate_relation_set(
    relations: Iterable[StateRelation],
    *,
    allow_empty: bool = False,
) -> None:
    """Validate relation semantics that depend on the whole upstream set.

    A single upstream state uses progress/branch/invalidate.  When the current
    state depends on two or more upstream states, each upstream edge is labeled
    combine.  Keeping one relation object per edge preserves direct state-ID
    lookup and graph construction.
    """
    relation_set = tuple(relations)
    if not relation_set:
        if allow_empty:
            return
        raise ValueError("state must declare at least one relation")

    if any(relation.type is StateRelationType.INIT for relation in relation_set):
        if len(relation_set) != 1 or relation_set[0].type is not StateRelationType.INIT:
            raise ValueError("init must be the only relation for a state with no upstream")
        return

    related_state_ids = [relation.related_state_id for relation in relation_set]
    if len(related_state_ids) != len(set(related_state_ids)):
        raise ValueError("relation set contains duplicate upstream state IDs")

    if len(relation_set) == 1:
        if relation_set[0].type is StateRelationType.COMBINE:
            raise ValueError("combine requires at least two related states")
        return

    if any(relation.type is not StateRelationType.COMBINE for relation in relation_set):
        raise ValueError(
            "two or more related states must be represented by one combine relation per state"
        )


@dataclass(frozen=True)
class AnalyticalState:
    """The full state object defined by the StateGuard method."""

    id: str
    issue: str
    constraints: tuple[Constraint, ...]
    used_variables: tuple[VariableRef, ...]
    conclusions: tuple[Conclusion, ...]
    relations: tuple[StateRelation, ...]
    source_step_start: int | None = None
    source_step_end: int | None = None
    checkpoint_id: str | None = None
    status: str = "committed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.issue.strip():
            raise ValueError("state id and issue must be non-empty")
        validate_relation_set(self.relations)
        if self.source_step_start is not None and self.source_step_end is not None:
            if self.source_step_start > self.source_step_end:
                raise ValueError("invalid source step span")
        variable_keys = [variable.key for variable in self.used_variables]
        if len(variable_keys) != len(set(variable_keys)):
            raise ValueError("state contains duplicate variable versions")
        wrong_versions = [
            variable.key for variable in self.used_variables if variable.version != self.id
        ]
        if wrong_versions:
            raise ValueError(
                f"state {self.id} contains variables not versioned by its state ID: "
                f"{wrong_versions}"
            )
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "issue": self.issue,
            "constraints": [item.to_dict() for item in self.constraints],
            "used_variables": [item.to_dict() for item in self.used_variables],
            "conclusions": [item.claim for item in self.conclusions],
            "relations": to_jsonable(self.relations),
            "source_step_start": self.source_step_start,
            "source_step_end": self.source_step_end,
            "checkpoint_id": self.checkpoint_id,
            "status": self.status,
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnalyticalState":
        required = {
            "id",
            "issue",
            "constraints",
            "used_variables",
            "conclusions",
            "relations",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(f"analytical state missing fields: {sorted(missing)}")
        return cls(
            id=str(value["id"]),
            issue=str(value["issue"]),
            constraints=tuple(Constraint.from_dict(x) for x in value["constraints"]),
            used_variables=tuple(VariableRef.from_dict(x) for x in value["used_variables"]),
            conclusions=tuple(Conclusion.from_value(x) for x in value["conclusions"]),
            relations=tuple(StateRelation.from_dict(x) for x in value["relations"]),
            source_step_start=value.get("source_step_start"),
            source_step_end=value.get("source_step_end"),
            checkpoint_id=value.get("checkpoint_id"),
            status=str(value.get("status", "committed")),
            metadata=dict(value.get("metadata", {})),
        )

    def as_state_hint(self, include_variables: bool = False) -> dict[str, Any]:
        """Minimal related-state observation injected into the worker."""
        hint: dict[str, Any] = {
            "id": self.id,
            "issue": self.issue,
            "conclusions": [item.claim for item in self.conclusions],
            "relations": [
                {"type": item.type.value, "related_state_id": item.related_state_id}
                for item in self.relations
            ],
        }
        if include_variables:
            kept = [
                entry
                for entry in (_hint_variable(item) for item in self.used_variables)
                if entry is not None
            ]
            if kept:
                hint["variables"] = kept
        return hint
