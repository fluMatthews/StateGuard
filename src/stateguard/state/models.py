from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any

from stateguard.core.models import to_jsonable


class StateRelationType(str, Enum):
    INIT = "init"
    PROGRESS = "progress"
    BRANCH = "branch"
    INVALIDATE = "invalidate"
    COMBINE = "combine"


@dataclass(frozen=True)
class Constraint:
    text: str
    source: str = "query"
    executable_predicate: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("constraint text must be non-empty")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Constraint":
        return cls(
            text=str(value["text"]),
            source=str(value.get("source", "query")),
            executable_predicate=value.get("executable_predicate"),
        )


@dataclass(frozen=True)
class VariableRef:
    name: str
    version: str
    value: Any = None
    value_summary: str = ""
    value_type: str = "unknown"
    producer_state_id: str | None = None
    producer_step_id: int | None = None
    artifact_ref: str | None = None
    digest: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("variable requires a non-empty name")
        if not isinstance(self.version, str) or not re.fullmatch(
            r"S[1-9][0-9]*", self.version
        ):
            raise ValueError("variable version must be the producing state ID, e.g. S3")
        if self.producer_state_id is not None and self.producer_state_id != self.version:
            raise ValueError("producer_state_id must match the variable's state-ID version")

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VariableRef":
        raw_version = str(value["version"]).strip()
        return cls(
            name=str(value["name"]),
            version=raw_version,
            value=value.get("value"),
            value_summary=str(value.get("value_summary", "")),
            value_type=str(value.get("value_type", "unknown")),
            producer_state_id=value.get("producer_state_id"),
            producer_step_id=value.get("producer_step_id"),
            artifact_ref=value.get("artifact_ref"),
            digest=value.get("digest"),
        )


@dataclass(frozen=True)
class Conclusion:
    id: str
    claim: str
    variable_keys: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.claim.strip():
            raise ValueError("conclusion id and claim must be non-empty")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Conclusion":
        return cls(
            id=str(value["id"]),
            claim=str(value["claim"]),
            variable_keys=tuple(str(x) for x in value.get("variable_keys", [])),
            evidence_refs=tuple(str(x) for x in value.get("evidence_refs", [])),
        )


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


@dataclass(frozen=True)
class AnalyticalState:
    """The full state object defined by the StateGuard method."""

    id: str
    issue: str
    confidence: float
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
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("state confidence must be in [0, 1]")
        if not self.relations:
            raise ValueError("state must declare at least one relation")
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
        known_variable_keys = set(variable_keys)
        unknown_conclusion_keys = sorted(
            {
                key
                for conclusion in self.conclusions
                for key in conclusion.variable_keys
                if key not in known_variable_keys
            }
        )
        if unknown_conclusion_keys:
            raise ValueError(
                "state conclusions reference variables absent from this state: "
                f"{unknown_conclusion_keys}"
            )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnalyticalState":
        required = {
            "id",
            "issue",
            "confidence",
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
            confidence=float(value["confidence"]),
            constraints=tuple(Constraint.from_dict(x) for x in value["constraints"]),
            used_variables=tuple(VariableRef.from_dict(x) for x in value["used_variables"]),
            conclusions=tuple(Conclusion.from_dict(x) for x in value["conclusions"]),
            relations=tuple(StateRelation.from_dict(x) for x in value["relations"]),
            source_step_start=value.get("source_step_start"),
            source_step_end=value.get("source_step_end"),
            checkpoint_id=value.get("checkpoint_id"),
            status=str(value.get("status", "committed")),
            metadata=dict(value.get("metadata", {})),
        )

    def as_state_hint(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "issue": self.issue,
            "used_variables": {
                item.key: item.value if item.value is not None else item.value_summary
                for item in self.used_variables
            },
            "conclusions": [item.claim for item in self.conclusions],
            "relations": [
                {"type": item.type.value, "related_state_id": item.related_state_id}
                for item in self.relations
            ],
        }

    def as_relation_view(self) -> dict[str, Any]:
        """Compact state content used for relation reasoning, not only lookup."""
        return {
            "id": self.id,
            "issue": self.issue,
            "confidence": self.confidence,
            "constraints": [
                {
                    "text": item.text,
                    "source": item.source,
                    "executable_predicate": item.executable_predicate,
                }
                for item in self.constraints
            ],
            "variables": [
                {
                    "key": item.key,
                    "value": item.value,
                    "value_type": item.value_type,
                    "value_summary": item.value_summary,
                    "producer_state_id": item.producer_state_id,
                    "producer_step_id": item.producer_step_id,
                    "artifact_ref": item.artifact_ref,
                    "digest": item.digest,
                }
                for item in self.used_variables
            ],
            "conclusions": [
                {
                    "id": item.id,
                    "claim": item.claim,
                    "variable_keys": list(item.variable_keys),
                    "evidence_refs": list(item.evidence_refs),
                }
                for item in self.conclusions
            ],
            "relations": [
                {"type": item.type.value, "related_state_id": item.related_state_id}
                for item in self.relations
            ],
            "status": self.status,
        }
