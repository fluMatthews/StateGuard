from __future__ import annotations

from stateguard.core.events import PendingInterval

from .models import ValidationFinding


class DeterministicValidator:
    """Optional high-precision findings library for manager-facing adapters.

    StateGuardHarness never invokes this automatically or converts findings into
    policy. A manager/tool adapter may call it when the manager chooses to check.
    """

    def inspect(self, interval: PendingInterval) -> tuple[ValidationFinding, ...]:
        findings: list[ValidationFinding] = []
        for step in interval.steps:
            if step.action.kind == "tool" and step.observation is None:
                findings.append(
                    ValidationFinding(
                        category="nl_fake_code",
                        message="The worker claimed a tool action but no execution observation exists.",
                        evidence=(f"step {step.step_id} has no tool result",),
                        step_ids=(step.step_id,),
                    )
                )
            if step.observation is not None and not step.observation.ok:
                findings.append(
                    ValidationFinding(
                        category="execution_error",
                        message="The executed tool call failed.",
                        evidence=(step.observation.error or "tool returned ok=false",),
                        step_ids=(step.step_id,),
                    )
                )
            if step.action.kind == "tool" and step.action.tool_name in {"python", "python_exec", "probe"}:
                code = step.action.arguments.get("code")
                if isinstance(code, str):
                    try:
                        compile(code, f"<worker-step-{step.step_id}>", "exec")
                    except SyntaxError as exc:
                        findings.append(
                            ValidationFinding(
                                category="syntax_error",
                                message="Executed Python is syntactically invalid.",
                                evidence=(f"{exc.msg} at line {exc.lineno}",),
                                step_ids=(step.step_id,),
                            )
                        )
            if step.done and not (step.action.answer or "").strip():
                findings.append(
                    ValidationFinding(
                        category="final_requirement",
                        message="The worker ended with an empty final answer.",
                        evidence=(f"step {step.step_id} final answer is empty",),
                        step_ids=(step.step_id,),
                    )
                )
        return tuple(findings)
