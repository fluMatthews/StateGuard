from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import Enum
from typing import Any

from stateguard.state.draft import (
    RelationFinalization,
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
    ABANDON_STATE = "ABANDON_STATE"
    ABSTAIN = "ABSTAIN"


ERROR_HINT_PROMPT = (
    "The Manager identified a suspected error in the listed variables or conclusions. "
    "Review the reason below and re-check the affected analysis; this hint is only for reference."
)


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
        if self.prompt.strip() != ERROR_HINT_PROMPT:
            raise ValueError("error hint prompt must use the fixed reference-only template")
        if len(self.faulty_reasoning) > 2000:
            raise ValueError("faulty_reasoning must stay local and evidence-focused")
        if any(
            marker in self.faulty_reasoning.lower()
            for marker in ("```python", "<python>", "<answer>")
        ):
            raise ValueError("error hint may localize an error but may not provide an executable/full answer")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ErrorHint":
        variables = value.get("error_variable", [])
        if isinstance(variables, str):
            variables = [variables]
        return cls(
            prompt=ERROR_HINT_PROMPT,
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
            "</error_hint>"
        )


@dataclass(frozen=True)
class AnalyticalEvidence:
    violated_constraints: tuple[str, ...]
    evidence: tuple[str, ...]
    suspected_state_ids: InitVar[tuple[str, ...] | None] = None
    suspected_step_ids: tuple[int, ...] = ()

    def __post_init__(
        self,
        suspected_state_ids: tuple[str, ...] | None,
    ) -> None:
        del suspected_state_ids
        if self.violated_constraints and not self.evidence:
            raise ValueError("a violated constraint requires concrete evidence")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnalyticalEvidence":
        constraints = value.get("violated_constraints", value.get("violated_constraint", []))
        if isinstance(constraints, str):
            constraints = [constraints]
        return cls(
            violated_constraints=tuple(str(x) for x in constraints),
            evidence=tuple(str(x) for x in value.get("evidence", [])),
            suspected_step_ids=tuple(int(x) for x in value.get("suspected_step_ids", [])),
        )


@dataclass(frozen=True)
class CleanupPlan:
    """Deprecated constructor compatibility; harness derives cleanup from hint."""

    remove_variables: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManagerDecision:
    action: ManagerAction
    state_header: StateHeader | None = None
    state_update: StateUpdate | None = None
    relation_finalization: RelationFinalization | None = None
    evidence: AnalyticalEvidence | None = None
    error_hint: ErrorHint | None = None
    cleanup: InitVar[CleanupPlan | None] = None

    def __post_init__(
        self,
        cleanup: CleanupPlan | None,
    ) -> None:
        if self.action is ManagerAction.OPEN_STATE and self.state_header is None:
            raise ValueError("OPEN_STATE requires a manager-written state_header")
        if self.action is ManagerAction.UPDATE_STATE and self.state_update is None:
            raise ValueError("UPDATE_STATE requires a state_update")
        if (
            self.action is ManagerAction.FINALIZE_RELATIONS
            and self.relation_finalization is None
        ):
            raise ValueError("FINALIZE_RELATIONS requires relation_finalization")
        if self.action is ManagerAction.REPAIR:
            if self.evidence is None or not self.evidence.evidence:
                raise ValueError("repair requires evidence-grounded localization")
            if not self.evidence.violated_constraints:
                raise ValueError("repair evidence requires an explicit violated constraint")
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
            state_header=StateHeader.from_dict(header_value) if header_value else None,
            state_update=StateUpdate.from_dict(update_value) if update_value is not None else None,
            relation_finalization=(
                RelationFinalization.from_dict(relation_value) if relation_value else None
            ),
            evidence=AnalyticalEvidence.from_dict(evidence_value) if evidence_value else None,
            error_hint=ErrorHint.from_dict(value["error_hint"]) if value.get("error_hint") else None,
        )


ManagerCommand = ManagerDecision


@dataclass(frozen=True)
class ValidationFinding:
    category: str
    message: str
    evidence: tuple[str, ...]
    step_ids: tuple[int, ...] = ()
