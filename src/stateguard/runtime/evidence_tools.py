from __future__ import annotations

import ast

from stateguard.core.models import ToolResult
from stateguard.harness.blind_view import assert_blind
from stateguard.state.store import StateStore

from .executors import CodeExecutor
from .tools import FunctionTool, ToolRegistry
from .trace import TraceBuffer
from .workspace import Workspace


def build_manager_evidence_tools(
    *,
    state_store: StateStore,
    workspace: Workspace,
    trace_buffer: TraceBuffer | None = None,
    probe_executor: CodeExecutor | None = None,
) -> ToolRegistry:
    """Create a read-only manager action space over committed evidence."""

    def load_state_index() -> dict:
        payload = {"states": state_store.load_state_index_json()}
        assert_blind(payload)
        return payload

    def load_state(state_id: str) -> dict:
        payload = state_store.load_state_json(state_id)
        assert_blind(payload)
        return payload

    tools = [
        FunctionTool(
            "load_state_index",
            "Load the compact committed-state index (id, issue, conclusions) for relation selection. This is a read load, not search.",
            load_state_index,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        FunctionTool(
            "load_state",
            "Load one exact committed-state JSON artifact by state ID for direct relation-state checking.",
            load_state,
            parameters={
                "type": "object",
                "properties": {
                    "state_id": {"type": "string", "pattern": "^S[1-9][0-9]*$"}
                },
                "required": ["state_id"],
                "additionalProperties": False,
            },
        ),
        FunctionTool(
            "compile_python",
            "Check Python syntax without executing it.",
            lambda code: {"syntax_ok": _compile_python(code)},
            parameters=_python_code_parameters(),
        ),
        FunctionTool(
            "inspect_python",
            "Inspect Python syntax plus assigned and loaded variable names without executing it.",
            _inspect_python,
            parameters=_python_code_parameters(),
        ),
    ]
    if trace_buffer is not None:

        def check_execution(step_id: int) -> dict:
            record = trace_buffer.get(step_id)
            step = record.step
            observation = step.observation
            native_tool_action = step.action.kind == "tool" and bool(
                step.action.tool_name
            )
            # Some official runtimes (notably smolagents CodeAgent) execute a
            # final_answer(...) Python action and return the answer in the same
            # native step. The core-facing action is terminal, but the matching
            # observation still records that real execution took place.
            observation_reports_execution = bool(
                observation is not None
                and observation.data.get("execution_attempted", False)
            )
            is_tool_call = native_tool_action or observation_reports_execution
            tool_name = (
                step.action.tool_name
                if native_tool_action
                else observation.tool_name if observation_reports_execution else None
            )
            submitted = dict(step.action.arguments) if native_tool_action else {}
            if observation_reports_execution and observation is not None:
                code = observation.data.get("code")
                if code:
                    submitted["code"] = code
            has_matching_result = bool(
                is_tool_call
                and observation is not None
                and observation.tool_name == tool_name
            )
            attempted = bool(
                is_tool_call
                and (
                    observation is None
                    or observation.data.get("execution_attempted", True)
                )
            )
            succeeded = bool(attempted and has_matching_result and observation.ok)
            if not is_tool_call or not attempted:
                status = "not_executed"
            elif not has_matching_result:
                status = "missing_result"
            elif observation.ok:
                status = "succeeded"
            else:
                status = "failed"
            return {
                "step_id": step_id,
                "status": status,
                "tool_call": is_tool_call,
                "tool_name": tool_name,
                "action_submitted": attempted,
                "matching_tool_result": has_matching_result,
                "execution_succeeded": succeeded,
                "arguments": submitted,
                "output": observation.output if has_matching_result else "",
                "error": observation.error if has_matching_result else None,
            }

        tools.append(
            FunctionTool(
                "check_execution",
                "Report whether one exact Worker native tool action received its matching executor result and whether that result succeeded. This covers Python, Bash, SQL, IPython, file actions, and other benchmark tools; it reports execution facts, not analytical correctness.",
                check_execution,
                parameters={
                    "type": "object",
                    "properties": {"step_id": {"type": "integer", "minimum": 1}},
                    "required": ["step_id"],
                    "additionalProperties": False,
                },
            )
        )
    if probe_executor is not None:

        def run_probe(code: str) -> ToolResult:
            result = probe_executor.execute(code)
            return ToolResult(
                "run_probe",
                result.ok,
                result.stdout,
                data={"stderr": result.stderr},
                error=result.error,
            )

        tools.append(
            FunctionTool(
                "run_probe",
                "Execute one code check in a fresh isolated scratch workspace. "
                "Worker analytical variables are not copied. You can load real task "
                "data through data_files and just run an independent probe to check the code.",
                run_probe,
                parameters=_python_code_parameters(),
            )
        )
    return ToolRegistry(
        tools,
        bindings={
            "state_store": state_store,
            "workspace": workspace,
            "trace_buffer": trace_buffer,
        },
    )


def _compile_python(code: str) -> bool:
    compile(code, "<manager-syntax-check>", "exec")
    return True


def _python_code_parameters() -> dict:
    """Strict shared schema for manager tools that accept Python source."""
    return {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
        "additionalProperties": False,
    }


def _inspect_python(code: str) -> dict:
    tree = ast.parse(code, filename="<manager-code-inspection>", mode="exec")
    assigned = sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
    )
    loaded = sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
    )
    return {"syntax_ok": True, "assigned_names": assigned, "loaded_names": loaded}
