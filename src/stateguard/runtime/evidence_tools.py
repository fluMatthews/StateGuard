from __future__ import annotations

import ast

from stateguard.core.models import ToolResult
from stateguard.state.store import StateStore

from .executors import IsolatedProbeExecutor
from .tools import FunctionTool, ToolRegistry
from .workspace import Workspace


def build_manager_evidence_tools(
    *,
    state_store: StateStore,
    workspace: Workspace,
    probe_executor: IsolatedProbeExecutor | None = None,
) -> ToolRegistry:
    """Create a read-only manager action space over committed evidence."""

    def list_state_index() -> dict:
        return {"states": state_store.index()}

    def get_state(state_id: str) -> dict:
        return state_store.get(state_id).to_dict()

    def read_relation_states(state_id: str) -> dict:
        direct_ids = list(state_store.relation_ids(state_id))
        return {
            "state_id": state_id,
            "direct_relation_ids": direct_ids,
            "max_upward_hops": 1,
            "states": [state_store.get(item).to_dict() for item in direct_ids],
        }

    def trace_variable(name_or_key: str) -> dict:
        matches = state_store.variable(name_or_key)
        return {
            "matches": [
                {"state_id": state_id, "variable": variable}
                for state_id, variable in matches
            ]
        }

    def trace_relations(state_id: str) -> dict:
        return {
            "direct_related_state_ids": state_store.relation_ids(state_id),
            "max_upward_hops": 1,
        }

    def workspace_manifest() -> dict:
        return workspace.manifest()

    tools = [
        FunctionTool(
            "list_state_index",
            "List committed state IDs, issues, variable keys, conclusions, and relation IDs.",
            list_state_index,
        ),
        FunctionTool("read_state", "Read one committed analytical state by exact ID.", get_state),
        FunctionTool(
            "read_relation_states",
            "Read only the direct one-hop upstream states named by relation IDs.",
            read_relation_states,
        ),
        FunctionTool(
            "trace_variable",
            "Trace a variable name or name@state_id through committed states.",
            trace_variable,
        ),
        FunctionTool(
            "trace_relations",
            "Read only direct one-hop upstream state relations.",
            trace_relations,
        ),
        FunctionTool(
            "workspace_manifest",
            "Read a non-mutating summary of the worker workspace.",
            workspace_manifest,
        ),
        FunctionTool(
            "compile_python",
            "Check Python syntax without executing it.",
            lambda code: {"ok": _compile_python(code)},
        ),
        FunctionTool(
            "inspect_python",
            "Inspect Python syntax plus assigned and loaded variable names without executing it.",
            _inspect_python,
        ),
    ]
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
                "Execute an assertion/check in an isolated clone; it cannot mutate worker state.",
                run_probe,
            )
        )
    return ToolRegistry(
        tools,
        bindings={"state_store": state_store, "workspace": workspace},
    )


def _compile_python(code: str) -> bool:
    compile(code, "<manager-syntax-check>", "exec")
    return True


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
