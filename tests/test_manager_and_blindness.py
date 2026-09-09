import json
import tempfile
import unittest
from pathlib import Path

from stateguard.agents.manager import StateManagerAgent
from stateguard.core.events import ReActStep
from stateguard.core.models import AgentAction, Message, TaskSpec, ToolResult
from stateguard.harness.blind_view import BlindViewBuilder, assert_blind
from stateguard.providers.base import ModelResponse, ScriptedModelClient
from stateguard.runtime.evidence_tools import build_manager_evidence_tools
from stateguard.runtime.executors import IsolatedProbeExecutor
from stateguard.runtime.workspace import InMemoryWorkspace
from stateguard.runtime.bundle import StateGuardRuntime
from stateguard.runtime.trace import TraceBuffer
from stateguard.agents.worker import WorkerAgent
from stateguard.state.store import StateStore
from stateguard.state.models import AnalyticalState, Constraint, StateRelation, StateRelationType
from stateguard.validation.models import ManagerAction


class ManagerAndBlindnessTest(unittest.TestCase):
    def test_manager_retries_one_transport_error_without_polluting_session(self):
        class FlakyModelClient:
            def __init__(self, response):
                self.response = response
                self.calls = 0

            def complete(self, messages, tools):
                del messages, tools
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("temporary read timeout")
                return ModelResponse(self.response)

        decision = {
            "action": "OPEN_STATE",
            "state_header": {
                "id": "S1",
                "issue": "Record the result",
                "constraints": [{"text": "Use checked evidence."}],
                "relations": [{"type": "init"}],
            },
        }
        model = FlakyModelClient(
            json.dumps({"type": "control", "answer": decision})
        )
        manager = StateManagerAgent(model)
        task = TaskSpec("transport-retry", "Record the result.")
        manager.start_task(task)
        request = BlindViewBuilder().build(
            task=task,
            event_type="TASK_START",
            flow_policy={"mode": "turn", "relation_timing": "query_first"},
            available_state_id="S1",
            worker_step=None,
            untraced_steps=(),
            current_draft=None,
            workspace_manifest={},
            repair_attempts=0,
        )

        result = manager.act(request)

        self.assertEqual(result.action, ManagerAction.OPEN_STATE)
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(manager.invocations), 2)
        self.assertTrue(manager.invocations[0]["will_retry"])
        self.assertIn("transport_error", manager.invocations[0])
        self.assertIn("command", manager.invocations[1])
        self.assertFalse(
            any("manager_transport_error" in message.content for message in manager.messages)
        )

    def test_manager_fails_after_second_transport_error(self):
        class FailingModelClient:
            def __init__(self):
                self.calls = 0

            def complete(self, messages, tools):
                del messages, tools
                self.calls += 1
                raise TimeoutError("persistent read timeout")

        model = FailingModelClient()
        manager = StateManagerAgent(model)
        task = TaskSpec("transport-failure", "Record the result.")
        manager.start_task(task)
        request = BlindViewBuilder().build(
            task=task,
            event_type="TASK_START",
            flow_policy={"mode": "turn", "relation_timing": "query_first"},
            available_state_id="S1",
            worker_step=None,
            untraced_steps=(),
            current_draft=None,
            workspace_manifest={},
            repair_attempts=0,
        )

        with self.assertRaises(TimeoutError):
            manager.act(request)

        self.assertEqual(model.calls, 2)
        self.assertEqual(len(manager.invocations), 3)
        self.assertTrue(manager.invocations[0]["will_retry"])
        self.assertFalse(manager.invocations[1]["will_retry"])
        self.assertIn("runtime_error", manager.invocations[2])

    def test_manager_session_persists_across_task_units(self):
        manager = StateManagerAgent(ScriptedModelClient([]))
        manager.start_task(TaskSpec("turn-1", "First turn."))
        first_messages = manager.messages
        manager.start_task(TaskSpec("turn-2", "Second turn."))
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
        self.assertIn("compile_python", tool_names)
        self.assertIn("inspect_python", tool_names)
        self.assertIn("check_execution", tool_names)
        self.assertIn("run_probe", tool_names)
        self.assertNotIn("load_state_index", tool_names)
        self.assertIn("load_state", tool_names)
        self.assertIs(runtime.manager, manager)

    def test_check_execution_reads_real_worker_tool_records(self):
        trace = TraceBuffer()
        trace.start_unit("unit-1")
        trace.append(
            ReActStep(
                step_id=1,
                action=AgentAction(
                    kind="tool",
                    reasoning="Run the count.",
                    tool_name="python",
                    arguments={"code": "print(3)"},
                ),
                observation=ToolResult("python", True, "3\n"),
                done=False,
            )
        )
        trace.append(
            ReActStep(
                step_id=2,
                action=AgentAction(
                    kind="final",
                    reasoning="I ran Python and obtained 3.",
                    answer="3",
                ),
                observation=None,
                done=True,
            )
        )
        trace.append(
            ReActStep(
                step_id=3,
                action=AgentAction(
                    kind="tool",
                    reasoning="Inspect the official task files.",
                    tool_name="execute_bash",
                    arguments={"command": "ls"},
                ),
                observation=ToolResult(
                    "execute_bash",
                    True,
                    "question.md\n",
                    {"execution_attempted": True, "execution_succeeded": True},
                ),
                done=False,
            )
        )
        trace.append(
            ReActStep(
                step_id=4,
                action=AgentAction(
                    kind="final",
                    reasoning="Return the executed Python result.",
                    answer="3",
                ),
                observation=ToolResult(
                    "python_interpreter",
                    True,
                    "",
                    {
                        "execution_attempted": True,
                        "execution_succeeded": True,
                        "code": "final_answer(3)",
                    },
                ),
                done=True,
            )
        )
        tools = build_manager_evidence_tools(
            state_store=StateStore(),
            workspace=InMemoryWorkspace(),
            trace_buffer=trace,
        )

        executed = tools.execute("check_execution", {"step_id": 1})
        executed_from_string = tools.execute("check_execution", {"step_id": "1"})
        claimed_only = tools.execute("check_execution", {"step_id": 2})
        native_bash = tools.execute("check_execution", {"step_id": 3})
        native_final = tools.execute("check_execution", {"step_id": 4})

        self.assertTrue(executed.ok)
        self.assertEqual(executed.data["status"], "succeeded")
        self.assertTrue(executed.data["execution_succeeded"])
        self.assertTrue(executed_from_string.ok)
        self.assertEqual(executed_from_string.data, executed.data)
        self.assertTrue(claimed_only.ok)
        self.assertEqual(claimed_only.data["status"], "not_executed")
        self.assertFalse(claimed_only.data["execution_succeeded"])
        self.assertEqual(native_bash.data["status"], "succeeded")
        self.assertEqual(native_bash.data["tool_name"], "execute_bash")
        self.assertEqual(native_bash.data["arguments"], {"command": "ls"})
        self.assertEqual(native_final.data["status"], "succeeded")
        self.assertEqual(native_final.data["tool_name"], "python_interpreter")
        self.assertEqual(native_final.data["arguments"], {"code": "final_answer(3)"})

    def test_stored_relation_ids_are_limited_to_direct_upstream_states(self):
        store = StateStore()
        states = (
            AnalyticalState(
                id="S1",
                issue="initial",
                constraints=(Constraint("initial constraint"),),
                used_variables=(),
                conclusions=(),
                relations=(StateRelation(StateRelationType.INIT),),
            ),
            AnalyticalState(
                id="S2",
                issue="progress",
                constraints=(Constraint("progress constraint"),),
                used_variables=(),
                conclusions=(),
                relations=(StateRelation(StateRelationType.PROGRESS, "S1"),),
            ),
            AnalyticalState(
                id="S3",
                issue="more progress",
                constraints=(Constraint("more progress constraint"),),
                used_variables=(),
                conclusions=(),
                relations=(StateRelation(StateRelationType.PROGRESS, "S2"),),
            ),
        )
        for state in states:
            store.commit(state)
        self.assertEqual(store.relation_ids("S3"), ("S2",))

    def test_react_manager_returns_open_state_command(self):
        decision = {
            "action": "OPEN_STATE",
            "state_header": {
                "id": "S1",
                "issue": "Return an executed result",
                "constraints": [{"text": "Use the observed execution result."}],
                "relations": [{"type": "init"}],
            },
        }
        model_output = json.dumps({"type": "control", "answer": decision})
        manager = StateManagerAgent(ScriptedModelClient([model_output]))
        task = TaskSpec("t", "Return an executed result.")
        manager.start_task(task)
        request = BlindViewBuilder().build(
            task=task,
            event_type="TASK_START",
            flow_policy={"mode": "turn", "relation_timing": "query_first"},
            available_state_id="S1",
            worker_step=None,
            untraced_steps=(),
            current_draft=None,
            workspace_manifest={},
            repair_attempts=0,
        )
        result = manager.act(request)
        self.assertEqual(result.action, ManagerAction.OPEN_STATE)
        self.assertEqual(result.state_header.id, "S1")
        self.assertEqual(len(manager.invocations), 1)

    def test_manager_retries_one_legacy_final_protocol_response(self):
        decision = {
            "action": "OPEN_STATE",
            "state_header": {
                "id": "S1",
                "issue": "Record the result",
                "constraints": [{"text": "Use checked evidence."}],
                "relations": [{"type": "init"}],
            },
        }
        manager = StateManagerAgent(
            ScriptedModelClient(
                [
                    json.dumps({"type": "final", "answer": decision}),
                    json.dumps({"type": "control", "answer": decision}),
                ]
            )
        )
        task = TaskSpec("retry", "Record the result.")
        manager.start_task(task)
        request = BlindViewBuilder().build(
            task=task,
            event_type="TASK_START",
            flow_policy={"mode": "turn", "relation_timing": "query_first"},
            available_state_id="S1",
            worker_step=None,
            untraced_steps=(),
            current_draft=None,
            workspace_manifest={},
            repair_attempts=0,
        )

        result = manager.act(request)

        self.assertEqual(result.action, ManagerAction.OPEN_STATE)
        self.assertEqual(len(manager.invocations), 2)
        self.assertIn("protocol_error", manager.invocations[0])
        self.assertTrue(
            any("<manager_protocol_error>" in message.content for message in manager.messages)
        )

    def test_forbidden_gt_key_is_blocked_recursively(self):
        with self.assertRaises(ValueError):
            assert_blind({"metadata": {"gold_answer": "secret"}})

    def test_probe_cannot_mutate_worker_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "values.txt"
            path.write_text("checked", encoding="utf-8")
            workspace = InMemoryWorkspace(
                variables={"x": 2},
                data_files={"values.txt": str(path)},
            )
            tools = build_manager_evidence_tools(
                state_store=StateStore(),
                workspace=workspace,
                probe_executor=IsolatedProbeExecutor(workspace),
            )
            result = tools.execute(
                "run_probe",
                {
                    "code": (
                        "assert 'x' not in globals()\n"
                        "with open(data_files['values.txt']) as handle:\n"
                        "    observed = handle.read()\n"
                        "x = 999\n"
                        "print(observed, x)"
                    )
                },
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.output.strip(), "checked 999")
        self.assertEqual(workspace.variables["x"], 2)
        self.assertNotIn("observed", workspace.variables)

    def test_manager_tools_expose_only_explicit_state_loads_not_search(self):
        store = StateStore()
        store.commit(
            AnalyticalState(
                id="S1",
                issue="Keep a checked value",
                constraints=(Constraint("Use checked evidence."),),
                used_variables=(),
                conclusions=(),
                relations=(StateRelation(StateRelationType.INIT),),
            )
        )
        tools = build_manager_evidence_tools(
            state_store=store,
            workspace=InMemoryWorkspace(),
        )
        tool_names = {schema["name"] for schema in tools.schemas()}
        self.assertNotIn("load_state_index", tool_names)
        self.assertIn("load_state", tool_names)
        self.assertNotIn("trace_variable", tool_names)
        self.assertNotIn("list_state_index", tool_names)
        self.assertNotIn("read_state", tool_names)
        self.assertNotIn("read_relation_states", tool_names)
        self.assertNotIn("trace_relations", tool_names)
        self.assertNotIn("workspace_manifest", tool_names)
        self.assertNotIn("search_states", tool_names)
        exact = tools.execute("load_state", {"state_id": "S1"})
        self.assertTrue(exact.ok)
        self.assertEqual(exact.data["id"], "S1")

    def test_manager_context_keeps_preamble_and_newest_complete_action_blocks(self):
        manager = StateManagerAgent(
            ScriptedModelClient([]),
            max_context_chars=None,
        )
        manager.start_task(TaskSpec("t", "Inspect the worker."))
        pinned_size = sum(len(message.content) for message in manager.messages[:2])
        manager.max_context_chars = pinned_size + 120
        manager.inject_observation(
            "OLD_OBSERVATION_" + "x" * 90,
            metadata={"manager_block_start": True},
        )
        manager.session.messages.append(Message("assistant", "OLD_TOOL_CALL"))
        manager.session.messages.append(Message("user", "OLD_TOOL_RESULT"))
        manager.inject_observation(
            "NEW_OBSERVATION",
            metadata={"manager_block_start": True},
        )
        manager.session.messages.append(Message("assistant", "NEW_TOOL_CALL"))
        manager.session.messages.append(Message("user", "NEW_TOOL_RESULT"))

        payload = manager._messages_for_model()
        contents = [message.content for message in payload]
        self.assertEqual(payload[:2], list(manager.messages[:2]))
        self.assertNotIn("OLD_TOOL_CALL", contents)
        self.assertIn("NEW_OBSERVATION", contents)
        self.assertIn("NEW_TOOL_CALL", contents)
        self.assertIn("NEW_TOOL_RESULT", contents)

    def test_python_evidence_tools_expose_strict_code_schemas(self):
        workspace = InMemoryWorkspace()
        tools = build_manager_evidence_tools(
            state_store=StateStore(),
            workspace=workspace,
            probe_executor=IsolatedProbeExecutor(workspace),
        )
        schemas = {schema["name"]: schema["parameters"] for schema in tools.schemas()}
        expected = {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        }
        for name in ("compile_python", "inspect_python", "run_probe"):
            self.assertEqual(schemas[name], expected)

    def test_compile_python_uses_syntax_ok_instead_of_duplicate_ok(self):
        tools = build_manager_evidence_tools(
            state_store=StateStore(),
            workspace=InMemoryWorkspace(),
        )
        result = tools.execute("compile_python", {"code": "x = 1"})
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"syntax_ok": True})
        self.assertNotIn("ok", result.data)


if __name__ == "__main__":
    unittest.main()
