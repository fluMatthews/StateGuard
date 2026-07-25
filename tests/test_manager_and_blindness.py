import json
import unittest

from stateguard.agents.manager import StateManagerAgent
from stateguard.core.models import TaskSpec
from stateguard.harness.blind_view import BlindViewBuilder, assert_blind
from stateguard.providers.base import ScriptedModelClient
from stateguard.runtime.evidence_tools import build_manager_evidence_tools
from stateguard.runtime.executors import IsolatedProbeExecutor
from stateguard.runtime.workspace import InMemoryWorkspace
from stateguard.runtime.bundle import StateGuardRuntime
from stateguard.agents.worker import WorkerAgent
from stateguard.state.store import StateStore
from stateguard.state.models import AnalyticalState, Constraint, StateRelation, StateRelationType
from stateguard.validation.models import ManagerAction


class ManagerAndBlindnessTest(unittest.TestCase):
    def test_manager_session_persists_across_task_units(self):
        manager = StateManagerAgent(ScriptedModelClient([]))
        manager.start_task(TaskSpec("turn-1", "First turn."), [])
        first_messages = manager.messages
        manager.start_task(TaskSpec("turn-2", "Second turn."), [])
        self.assertGreater(len(manager.messages), len(first_messages))
        self.assertTrue(
            any("First turn." in message.content for message in manager.messages)
        )
        self.assertIn("Second turn.", manager.messages[-1].content)

    def test_runtime_wires_manager_tools_to_its_own_evidence(self):
        manager = StateManagerAgent(ScriptedModelClient([]))
        runtime = StateGuardRuntime.create(
            worker=WorkerAgent(ScriptedModelClient([])),
            manager=manager,
            workspace=InMemoryWorkspace(),
        )
        tool_names = {schema["name"] for schema in manager.tools.schemas()}
        self.assertIn("read_state", tool_names)
        self.assertIn("inspect_python", tool_names)
        self.assertIn("run_probe", tool_names)
        self.assertIs(runtime.manager, manager)

    def test_relation_reads_are_limited_to_one_upstream_hop(self):
        store = StateStore()
        states = (
            AnalyticalState(
                id="S1",
                issue="initial",
                confidence=1.0,
                constraints=(Constraint("initial constraint"),),
                used_variables=(),
                conclusions=(),
                relations=(StateRelation(StateRelationType.INIT),),
            ),
            AnalyticalState(
                id="S2",
                issue="progress",
                confidence=1.0,
                constraints=(Constraint("progress constraint"),),
                used_variables=(),
                conclusions=(),
                relations=(StateRelation(StateRelationType.PROGRESS, "S1"),),
            ),
            AnalyticalState(
                id="S3",
                issue="more progress",
                confidence=1.0,
                constraints=(Constraint("more progress constraint"),),
                used_variables=(),
                conclusions=(),
                relations=(StateRelation(StateRelationType.PROGRESS, "S2"),),
            ),
        )
        for state in states:
            store.commit(state)
        tools = build_manager_evidence_tools(
            state_store=store,
            workspace=InMemoryWorkspace(),
        )
        result = tools.execute(
            "read_relation_states", {"state_id": "S3"}
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.data["max_upward_hops"], 1)
        self.assertEqual([state["id"] for state in result.data["states"]], ["S2"])

    def test_react_manager_returns_open_state_command(self):
        decision = {
            "action": "OPEN_STATE",
            "note": "Initialize the state before tracing.",
            "confidence": 0.95,
            "state_header": {
                "id": "S1",
                "issue": "Return an executed result",
                "constraints": [{"text": "Use the observed execution result."}],
                "relations": [{"type": "init"}],
            },
        }
        model_output = json.dumps({"type": "final", "answer": decision})
        manager = StateManagerAgent(ScriptedModelClient([model_output]))
        task = TaskSpec("t", "Return an executed result.")
        manager.start_task(task, [])
        request = BlindViewBuilder().build(
            task=task,
            event_type="TASK_START",
            flow_policy={"mode": "turn", "relation_timing": "query_first"},
            available_state_id="S1",
            worker_step=None,
            untraced_steps=(),
            current_draft=None,
            state_index=[],
            stored_states=[],
            relation_states=[],
            workspace_manifest={},
            repair_attempts=0,
        )
        result = manager.act(request)
        self.assertEqual(result.action, ManagerAction.OPEN_STATE)
        self.assertEqual(result.state_header.id, "S1")
        self.assertEqual(len(manager.invocations), 1)

    def test_forbidden_gt_key_is_blocked_recursively(self):
        with self.assertRaises(ValueError):
            assert_blind({"metadata": {"gold_answer": "secret"}})

    def test_probe_cannot_mutate_worker_workspace(self):
        workspace = InMemoryWorkspace(variables={"x": 2})
        tools = build_manager_evidence_tools(
            state_store=StateStore(),
            workspace=workspace,
            probe_executor=IsolatedProbeExecutor(workspace),
        )
        result = tools.execute("run_probe", {"code": "x = 999\nassert x == 999\nprint(x)"})
        self.assertTrue(result.ok)
        self.assertEqual(workspace.variables["x"], 2)

    def test_manager_state_tools_are_exact_id_based_not_search_based(self):
        tools = build_manager_evidence_tools(
            state_store=StateStore(),
            workspace=InMemoryWorkspace(),
        )
        tool_names = {schema["name"] for schema in tools.schemas()}
        self.assertIn("read_state", tool_names)
        self.assertIn("read_relation_states", tool_names)
        self.assertNotIn("search_states", tool_names)


if __name__ == "__main__":
    unittest.main()
