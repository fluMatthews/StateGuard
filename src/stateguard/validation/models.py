from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from stateguard.state.draft import (
    RelationFinalization,
    RelationFinalizationMode,
    StateHeader,
    StateUpdate,
)


class ManagerAction(str, Enum):
    RESUME_WORKER = "RESUME_WORKER"
    OPEN_STATE = "OPEN_STATE"
    UPDATE_STATE = "UPDATE_STATE"
    FINALIZE_RELATIONS = "FINALIZE_RELATIONS"
    COMMIT_STATE = "COMMIT_STATE"
    REPAIR = "REPAIR"
    ROLLBACK_PASS = "ROLLBACK_PASS"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class ErrorHint:
    prompt: str
    error_variable: tuple[str, ...]
    faulty_reasoning: str

    def __post_init__(self) -> None:
        if not self.prompt.strip() or not self.faulty_reasoning.strip():
            raise ValueError("error hint prompt and faulty_reasoning must be non-empty")
        if not self.error_variable:
            raise ValueError("error hint must identify at least one variable or conclusion")
        rendered = self.prompt + self.faulty_reasoning
        if len(rendered) > 4000:
            raise ValueError("error hint must be a local repair pointer, not a replacement solution")
        if any(marker in rendered.lower() for marker in ("```python", "<python>", "<answer>")):
            raise ValueError("error hint may localize an error but may not provide an executable/full answer")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ErrorHint":
        variables = value.get("error_variable", [])
        if isinstance(variables, str):
            variables = [variables]
        return cls(
            prompt=str(value["prompt"]),
            error_variable=tuple(str(x) for x in variables),
            faulty_reasoning=str(value["faulty_reasoning"]),
        )

    def as_observation(self) -> str:
        variables = ", ".join(self.error_variable)
        return (
            "<error_hint>\n"
            f"prompt: {self.prompt}\n"
            f"error_variable: {variables}\n"
            f"faulty_reasoning: {self.faulty_reasoning}\n"
            "</error_hint>\n"
            "Re-check the evidence and redo the affected analysis. Use tools to verify the repair."
        )


@dataclass(frozen=True)
class AnalyticalEvidence:
    confidence: float
    violated_constraints: tuple[str, ...]
    evidence: tuple[str, ...]
    suspected_state_ids: tuple[str, ...] = ()
    suspected_step_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("evidence confidence must be in [0, 1]")
        if self.violated_constraints and not self.evidence:
            raise ValueError("a violated constraint requires concrete evidence")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnalyticalEvidence":
        constraints = value.get("violated_constraints", value.get("violated_constraint", []))
        if isinstance(constraints, str):
            constraints = [constraints]
        return cls(
            confidence=float(value.get("confidence", 0.0)),
            violated_constraints=tuple(str(x) for x in constraints),
            evidence=tuple(str(x) for x in value.get("evidence", [])),
            suspected_state_ids=tuple(str(x) for x in value.get("suspected_state_ids", [])),
            suspected_step_ids=tuple(int(x) for x in value.get("suspected_step_ids", [])),
        )


@dataclass(frozen=True)
class CleanupPlan:
    remove_variables: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "CleanupPlan":
        value = value or {}
        return cls(
            remove_variables=tuple(str(x) for x in value.get("remove_variables", [])),
        )


@dataclass(frozen=True)
class ManagerDecision:
    action: ManagerAction
    note: str
    confidence: float = 0.0
    state_header: StateHeader | None = None
    state_update: StateUpdate | None = None
    relation_finalization: RelationFinalization | None = None
    evidence: AnalyticalEvidence | None = None
    error_hint: ErrorHint | None = None
    cleanup: CleanupPlan = field(default_factory=CleanupPlan)

    def __post_init__(self) -> None:
        if not self.note.strip():
            raise ValueError("manager decision requires a note")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("manager confidence must be in [0, 1]")
        if self.action is ManagerAction.OPEN_STATE and self.state_header is None:
            raise ValueError("OPEN_STATE requires a manager-written state_header")
        if self.action is ManagerAction.UPDATE_STATE and self.state_update is None:
            raise ValueError("UPDATE_STATE requires a state_update")
        if (
            self.action is ManagerAction.FINALIZE_RELATIONS
            and self.relation_finalization is None
        ):
            raise ValueError("FINALIZE_RELATIONS requires relation_finalization")
        if (
            self.action is ManagerAction.FINALIZE_RELATIONS
            and self.relation_finalization is not None
            and self.relation_finalization.mode is RelationFinalizationMode.RESELECT
            and self.confidence < 0.8
        ):
            raise ValueError(
                "changing a provisional relation requires high-confidence explicit conflict"
            )
        if self.action is ManagerAction.REPAIR:
            if self.confidence < 0.8:
                raise ValueError("repair requires high confidence (>= 0.8); ambiguity must not repair")
            if self.evidence is None or not self.evidence.evidence:
                raise ValueError("repair requires evidence-grounded localization")
            if self.evidence.confidence < 0.8 or not self.evidence.violated_constraints:
                raise ValueError("repair evidence requires high confidence and an explicit violated constraint")
            if self.error_hint is None:
                raise ValueError("repair requires a structured error hint")
            if any(
                item is not None
                for item in (
                    self.state_header,
                    self.state_update,
                    self.relation_finalization,
                )
            ):
                raise ValueError("a rejected repair branch cannot write state")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ManagerDecision":
        action = ManagerAction(str(value.get("action", value.get("decision", ""))).upper())
        header_value = value.get("state_header")
        update_value = value.get("state_update")
        relation_value = value.get("relation_finalization")
        evidence_value = value.get("analytical_evidence", value.get("evidence_bundle"))
        if evidence_value is None and isinstance(value.get("evidence"), dict):
            evidence_value = value["evidence"]
        return cls(
            action=action,
            note=str(value.get("note", "")),
            confidence=float(value.get("confidence", 0.0)),
            state_header=StateHeader.from_dict(header_value) if header_value else None,
            state_update=StateUpdate.from_dict(update_value) if update_value is not None else None,
            relation_finalization=(
                RelationFinalization.from_dict(relation_value) if relation_value else None
            ),
            evidence=AnalyticalEvidence.from_dict(evidence_value) if evidence_value else None,
            error_hint=ErrorHint.from_dict(value["error_hint"]) if value.get("error_hint") else None,
            cleanup=CleanupPlan.from_dict(value.get("cleanup")),
        )


ManagerCommand = ManagerDecision


@dataclass(frozen=True)
class ValidationFinding:
    category: str
    confidence: float
    message: str
    evidence: tuple[str, ...]
    step_ids: tuple[int, ...] = ()
