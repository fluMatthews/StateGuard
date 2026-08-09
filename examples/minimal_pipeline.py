"""A deterministic worker+manager run showing append-only repair and commit."""

from __future__ import annotations

import json

from stateguard.agents.manager import StateManagerAgent
from stateguard.agents.worker import WorkerAgent
from stateguard.core.models import TaskSpec
from stateguard.harness.engine import StateGuardHarness
from stateguard.providers.base import ScriptedModelClient
from stateguard.runtime.evidence_tools import build_manager_evidence_tools
from stateguard.runtime.executors import IsolatedProbeExecutor
from stateguard.runtime.workspace import InMemoryWorkspace
from stateguard.state.graph import StateRelationGraph
from stateguard.validation.models import ERROR_HINT_PROMPT
from stateguard.state.store import StateStore


def outer_final(decision: dict) -> str:
    return json.dumps({"type": "final", "reasoning": "Structured manager decision.", "answer": decision})


workspace = InMemoryWorkspace()
store = StateStore()
graph = StateRelationGraph()
worker = WorkerAgent(
    ScriptedModelClient(
        [
            json.dumps({"type": "final", "reasoning": "mental arithmetic", "answer": "41"}),
            json.dumps({"type": "final", "reasoning": "rechecked after hint", "answer": "42"}),
        ]
    )
)

manager_responses = [
    outer_final(
        {
            "action": "OPEN_STATE",
            "state_header": {
                "id": "S1",
                "issue": "Compute 6 * 7",
                "constraints": [{"text": "Return the value of 6 * 7."}],
                "relations": [{"type": "init", "related_state_id": None}]
            }
        }
    ),
    outer_final({"action": "RESUME_WORKER"}),
    outer_final(
        {
            "action": "REPAIR",
            "confidence": 0.99,
            "analytical_evidence": {
                "confidence": 0.99,
                "violated_constraints": ["Return the value of 6 * 7."],
                "evidence": ["The worker returned 41 without executed evidence."],
                "suspected_step_ids": [1]
            },
            "error_hint": {
                "prompt": ERROR_HINT_PROMPT,
                "error_variable": ["final_result"],
                "faulty_reasoning": "The value 41 is unsupported and inconsistent with 6 * 7."
            }
        }
    ),
    outer_final(
        {
            "action": "UPDATE_STATE",
            "state_update": {
                "used_variables": [{
                    "name": "final_result", "version": "S1", "value": 42
                }],
                "conclusions": ["The result is 42."],
                "traced_step_ids": [2]
            }
        }
    ),
    outer_final(
        {
            "action": "FINALIZE_RELATIONS",
            "relation_finalization": {
                "mode": "confirm",
                "relations": [{"type": "init", "related_state_id": None}],
                "reason": "There is no predecessor and the repaired trace satisfies the initial query."
            }
        }
    ),
    outer_final(
        {
            "action": "COMMIT_STATE",
            "confidence": 0.99
        }
    ),
]
manager = StateManagerAgent(
    ScriptedModelClient(manager_responses),
    build_manager_evidence_tools(
        state_store=store,
        workspace=workspace,
        probe_executor=IsolatedProbeExecutor(workspace),
    ),
)
harness = StateGuardHarness(
    worker=worker,
    manager=manager,
    workspace=workspace,
    state_store=store,
    graph=graph,
)
result = harness.run(TaskSpec("demo", "Return the value of 6 * 7."))
print(json.dumps({"final_answer": result.final_answer, "repairs": result.repair_count, "states": len(result.committed_states)}, indent=2))
