import json
import tempfile
import unittest
from pathlib import Path

from stateguard.adapters.flow import FixedStepFlowAdapter
from stateguard.agents.manager import StateManagerAgent
from stateguard.agents.worker import WorkerAgent
from stateguard.core.events import ReActStep
from stateguard.core.models import AgentAction, TaskSpec
from stateguard.harness.engine import StateGuardConfig, StateGuardHarness
from stateguard.providers.base import ScriptedModelClient
from stateguard.runtime.workspace import InMemoryWorkspace
from stateguard.runtime.bundle import StateGuardRuntime
from stateguard.runtime.executors import TrustedPythonExecutor
from stateguard.runtime.checkpoints import CheckpointManager
from stateguard.repair.controller import RepairController, RepairDirective, RepairSession
from stateguard.runtime.trace import TraceBuffer
from stateguard.runtime.tools import FunctionTool, ToolRegistry
from stateguard.state.models import (
    Conclusion,
    Constraint,
    StateRelation,
    StateRelationType,
    VariableRef,
)
from stateguard.state.draft import (
    DraftStore,
    RelationFinalization,
    RelationFinalizationMode,
    StateHeader,
    StateUpdate,
)
from stateguard.state.graph import StateRelationGraph
from stateguard.state.models import AnalyticalState
from stateguard.state.store import StateStore
from stateguard.validation.models import (
    AnalyticalEvidence,
    ERROR_HINT_PROMPT,
    CleanupPlan,
    ErrorHint,
    ManagerAction,
    ManagerDecision,
)


class RepairThenCommitManager:
    def __init__(self):
        self.state_updated = False
        self.relations_finalized = False
        self.commands = [
            ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    id="S1",
                    issue="Compute 6 * 7",
                    constraints=(Constraint("Return the value of 6 * 7."),),
                    relations=(StateRelation(StateRelationType.INIT),),
                ),
            ),
            ManagerDecision(
                action=ManagerAction.RESUME_WORKER,
            ),
            ManagerDecision(
                action=ManagerAction.REPAIR,
                confidence=0.99,
                evidence=AnalyticalEvidence(
                    confidence=0.99,
                    violated_constraints=("Return the executed value of 6 * 7.",),
                    evidence=("The worker answered 41 without an execution result.",),
                    suspected_step_ids=(1,),
                ),
                error_hint=ErrorHint(
                    prompt=ERROR_HINT_PROMPT,
                    error_variable=("final_result",),
                    faulty_reasoning="The answer 41 is unsupported and conflicts with 6 * 7.",
                ),
            ),
        ]

    def start_task(self, task):
        del task

    def act(self, request):
        if self.commands:
            return self.commands.pop(0)
        variable = VariableRef(
            "final_result",
            "S1",
            value=42,
        )
        if not self.state_updated:
            self.state_updated = True
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    used_variables=(variable,),
                    conclusions=(
                        Conclusion("The result is 42."),
                    ),
                    traced_step_ids=(request.worker_step.step_id,),
                ),
            )
        if not self.relations_finalized:
            self.relations_finalized = True
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    mode=RelationFinalizationMode.CONFIRM,
                    relations=(StateRelation(StateRelationType.INIT),),
                    reason="No predecessor exists and the checked turn remains task initialization.",
                ),
            )
        return ManagerDecision(
            action=ManagerAction.COMMIT_STATE,
            confidence=0.99,
        )


class SnapshotAwareRepairThenCommitManager(RepairThenCommitManager):
    def __init__(self):
        super().__init__()
        self.restore_calls = 0

    def snapshot(self):
        return {"remaining_commands": len(self.commands)}

    def restore(self, snapshot):
        del snapshot
        self.restore_calls += 1


class SparseRepairThenCommitManager:
    """After repair, keep only correct supporting steps from the rewritten interval."""

    def __init__(self):
        self.observations = []

    def start_task(self, task):
        del task

    def act(self, request):
        self.observations.append(request)
        index = len(self.observations)
        if index == 1:
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    id="S1",
                    issue="Produce a corrected supported result",
                    constraints=(Constraint("Use only correct rewritten evidence."),),
                    relations=(StateRelation(StateRelationType.INIT),),
                ),
            )
        if index == 2:
            return ManagerDecision(action=ManagerAction.RESUME_WORKER)
        if index == 3:
            return ManagerDecision(
                action=ManagerAction.REPAIR,
                confidence=0.99,
                evidence=AnalyticalEvidence(
                    confidence=0.99,
                    violated_constraints=("Use only correct rewritten evidence.",),
                    evidence=("The first answer is explicitly unsupported.",),
                    suspected_step_ids=(1,),
                ),
                error_hint=ErrorHint(
                    prompt=ERROR_HINT_PROMPT,
                    error_variable=("result",),
                    faulty_reasoning="The first answer has no supporting execution.",
                ),
            )
        if index == 4:
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    conclusions=(Conclusion("The rewritten result is supported."),),
                    # Step 3 is an irrelevant intermediate action inside the
                    # repaired interval and is deliberately excluded.
                    traced_step_ids=(2, 4, 5),
                ),
            )
        if index == 5:
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    mode=RelationFinalizationMode.CONFIRM,
                    relations=(StateRelation(StateRelationType.INIT),),
                    reason="The rewritten state remains the initial state.",
                ),
            )
        return ManagerDecision(action=ManagerAction.COMMIT_STATE)


class RecordingModelClient(ScriptedModelClient):
    def __init__(self, responses):
        super().__init__(responses)
        self.message_snapshots = []

    def complete(self, messages, tools):
        self.message_snapshots.append(tuple(messages))
        return super().complete(messages, tools)


class RelationFirstManager:
    """Open S2 with exact relation IDs before observing any S2 worker trace."""

    def __init__(self):
        self.observations = []

    def start_task(self, task):
        del task

    def act(self, request):
        self.observations.append(request)
        if len(self.observations) == 1:
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    id="S2",
                    issue="Report the prior result",
                    constraints=(Constraint("Return the value represented by S1."),),
                    relations=(
                        StateRelation(StateRelationType.PROGRESS, related_state_id="S1"),
                    ),
                ),
            )
        if len(self.observations) == 2:
            return ManagerDecision(
                action=ManagerAction.RESUME_WORKER,
            )
        if len(self.observations) == 3:
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    conclusions=(Conclusion("The reported value is 42."),),
                    traced_step_ids=(1,),
                ),
            )
        if len(self.observations) == 4:
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    mode=RelationFinalizationMode.CONFIRM,
                    relations=(StateRelation(StateRelationType.PROGRESS, "S1"),),
                    reason="The output directly reuses the value produced by S1.",
                ),
            )
        return ManagerDecision(
            action=ManagerAction.COMMIT_STATE,
            confidence=0.95,
        )


class RelationReselectManager:
    """Replace a provisional relation only after concrete current-state conflict."""

    def __init__(self):
        self.observations = []

    def start_task(self, task):
        del task

    def act(self, request):
        self.observations.append(request)
        index = len(self.observations)
        if index == 1:
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    id="S3",
                    issue="Identify the actual reused result",
                    constraints=(Constraint("Report the result actually used."),),
                    relations=(StateRelation(StateRelationType.PROGRESS, "S1"),),
                ),
            )
        if index == 2:
            return ManagerDecision(
                action=ManagerAction.RESUME_WORKER,
            )
        if index == 3:
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    conclusions=(Conclusion("The worker actually reused S2."),),
                    traced_step_ids=(1,),
                ),
            )
        if index == 4:
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                confidence=0.97,
                relation_finalization=RelationFinalization(
                    mode=RelationFinalizationMode.RESELECT,
                    relations=(StateRelation(StateRelationType.PROGRESS, "S2"),),
                    reason="Shared current-state/compact-index selection identifies S2.",
                    conflict_evidence=(
                        "Current conclusion names and reuses S2, while provisional S1 has a different result.",
                    ),
                ),
            )
        return ManagerDecision(
            action=ManagerAction.COMMIT_STATE,
        )


class FixedRepairSequenceManager:
    """Trigger repair only after inspecting each newly produced worker answer."""

    def __init__(self):
        self.observations = []

    def start_task(self, task):
        del task

    def act(self, request):
        self.observations.append(request)
        index = len(self.observations)
        if index == 1:
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    id="S1",
                    issue="Produce a verified answer",
                    constraints=(Constraint("The answer must be verified."),),
                    relations=(StateRelation(StateRelationType.INIT),),
                ),
            )
        if index == 2:
            return ManagerDecision(
                action=ManagerAction.RESUME_WORKER,
            )
        if index in {3, 4, 5}:
            return ManagerDecision(
                action=ManagerAction.REPAIR,
                confidence=0.99,
                evidence=AnalyticalEvidence(
                    0.99,
                    ("The answer must be verified.",),
                    (f"worker retry {index - 3} is explicitly marked unsupported",),
                    suspected_step_ids=(1,),
                ),
                error_hint=ErrorHint(
                    ERROR_HINT_PROMPT,
                    ("answer",),
                    "The latest worker output remains unsupported.",
                ),
            )
        return ManagerDecision(
            action=ManagerAction.ABANDON_STATE,
        )


class FixedWindowManager:
    def __init__(self):
        self.observations = []

    def start_task(self, task):
        del task

    def act(self, request):
        self.observations.append(request)
        if len(self.observations) == 1:
            return ManagerDecision(
                action=ManagerAction.RESUME_WORKER,
            )
        if len(self.observations) == 2:
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    id="S1",
                    constraints=(Constraint("Use the executed segment output."),),
                    relations=(),
                ),
            )
        if len(self.observations) == 3:
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    issue="Complete the five-step segment",
                    conclusions=(Conclusion("The five-step segment completed."),),
                    traced_step_ids=(1, 2, 3, 4, 5),
                ),
            )
        if len(self.observations) == 4:
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    mode=RelationFinalizationMode.SELECT,
                    relations=(StateRelation(StateRelationType.INIT),),
                    reason="No committed predecessor exists for this independent state.",
                ),
            )
        return ManagerDecision(
            action=ManagerAction.COMMIT_STATE,
            confidence=0.95,
        )


class ArbitraryBoundaryManager:
    """Choose a three-step state at a five-step mechanical review point."""

    def __init__(self):
        self.observations = []

    def start_task(self, task):
        del task

    def act(self, request):
        self.observations.append(request)
        index = len(self.observations)
        if index == 1:
            return ManagerDecision(
                action=ManagerAction.RESUME_WORKER,
            )
        if index == 2:
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    id="S1",
                    constraints=(Constraint("Use the selected executed prefix."),),
                    relations=(),
                ),
            )
        if index == 3:
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    issue="Record the first completed sub-result",
                    conclusions=(Conclusion("The first sub-result is complete."),),
                    traced_step_ids=(1, 2, 3),
                ),
            )
        if index == 4:
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    mode=RelationFinalizationMode.SELECT,
                    relations=(StateRelation(StateRelationType.INIT),),
                    reason="This is the first committed analytical state.",
                ),
            )
        if index == 5:
            return ManagerDecision(
                action=ManagerAction.COMMIT_STATE,
            )
        if index == 6:
            return ManagerDecision(
                action=ManagerAction.RESUME_WORKER,
            )
        return ManagerDecision(
            action=ManagerAction.ABSTAIN,
        )


class CrashingManager:
    def start_task(self, task):
        del task

    def act(self, request):
        del request
        raise RuntimeError("manager backend unavailable")


class CountingWorkspace(InMemoryWorkspace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.restore_calls = 0

    def restore(self, snapshot):
        self.restore_calls += 1
        super().restore(snapshot)


class CountingTraceBuffer(TraceBuffer):
    def __init__(self):
        super().__init__()
        self.restore_calls = 0

    def restore(self, snapshot):
        self.restore_calls += 1
        super().restore(snapshot)


class FailingUpdateDraftStore(DraftStore):
    def update(self, update):
        del update
        raise RuntimeError("injected draft update failure")


class FailingGraph(StateRelationGraph):
    def add_state(self, state):
        super().add_state(state)
        raise RuntimeError("injected graph write failure")


class SequenceManager:
    def __init__(self, commands):
        self.commands = list(commands)

    def start_task(self, task):
        del task

    def act(self, request):
        del request
        return self.commands.pop(0)


class HarnessTest(unittest.TestCase):
    def test_worker_python_tool_executes_against_real_staged_data_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_path = Path(temporary) / "numbers.csv"
            data_path.write_text("value\n1\n2\n", encoding="utf-8")
            model = ScriptedModelClient(
                [
                    json.dumps(
                        {
                            "type": "tool",
                            "reasoning": "Read and calculate from the real file.",
                            "tool": "python",
                            "arguments": {
                                "code": (
                                    "import csv\n"
                                    "with open(data_files['numbers.csv']) as handle:\n"
                                    "    total = sum(int(row['value']) for row in "
                                    "csv.DictReader(handle))\n"
                                    "print(total)"
                                )
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "final",
                            "reasoning": "Used the executed Python observation.",
                            "answer": "3",
                        }
                    ),
                ]
            )
            workspace = InMemoryWorkspace()
            worker = WorkerAgent(model)
            result = StateGuardHarness(
                worker=worker,
                manager=None,
                workspace=workspace,
            ).run(
                TaskSpec(
                    "real-file",
                    "Sum the value column.",
                    data_files=(str(data_path),),
                )
            )
            checkpoints = CheckpointManager({"workspace": workspace})
            executed = checkpoints.capture("after-python")
            workspace.variables["total"] = 999
            checkpoints.restore(executed)

        self.assertEqual(result.final_answer, "3")
        self.assertIn("python", worker.tools)
        self.assertEqual(workspace.variables["total"], 3)
        self.assertEqual(workspace.data_files["numbers.csv"], str(data_path.resolve()))
        self.assertTrue(
            any(
                message.name == "python" and '"output": "3\\n"' in message.content
                for message in worker.messages
            )
        )

    def test_manager_failure_is_recorded_and_worker_fails_open(self):
        worker = WorkerAgent(
            ScriptedModelClient(
                [json.dumps({"type": "final", "answer": "worker-answer", "reasoning": "done"})]
            )
        )
        workspace = CountingWorkspace()
        result = StateGuardHarness(
            worker=worker,
            manager=CrashingManager(),
            workspace=workspace,
        ).run(TaskSpec("manager-failure", "Answer with the worker."))

        self.assertEqual(result.final_answer, "worker-answer")
        self.assertTrue(result.degraded)
        self.assertGreaterEqual(len(result.manager_failures), 1)
        self.assertEqual(result.manager_failures[0].error_type, "RuntimeError")
        self.assertEqual(workspace.restore_calls, 0)

    def test_two_malformed_manager_outputs_cancel_intervention(self):
        worker = WorkerAgent(
            ScriptedModelClient(
                [json.dumps({"type": "final", "answer": "worker-answer", "reasoning": "done"})]
            )
        )
        manager = StateManagerAgent(
            ScriptedModelClient(["not-json-once", "not-json-twice"])
        )
        workspace = CountingWorkspace()

        result = StateGuardHarness(
            worker=worker,
            manager=manager,
            workspace=workspace,
        ).run(TaskSpec("malformed-manager", "Let the worker answer."))

        self.assertEqual(result.final_answer, "worker-answer")
        self.assertTrue(result.degraded)
        self.assertEqual(len(manager.invocations), 2)
        self.assertEqual(workspace.restore_calls, 0)

    def test_checkpoint_can_restore_only_selected_control_components(self):
        worker = WorkerAgent(ScriptedModelClient([]))
        worker.start("original")
        workspace = InMemoryWorkspace(variables={"value": 1})
        checkpoints = CheckpointManager({"worker": worker, "workspace": workspace})
        selected = checkpoints.capture("worker-only", components=("worker",))
        worker.inject_observation("temporary")
        workspace.variables["value"] = 2

        checkpoints.restore(selected)

        self.assertFalse(any(message.content == "temporary" for message in worker.messages))
        self.assertEqual(workspace.variables["value"], 2)

    def test_definite_preflight_error_gets_one_manager_correction(self):
        worker = WorkerAgent(
            ScriptedModelClient(
                [json.dumps({"type": "final", "answer": "done", "reasoning": "done"})]
            )
        )
        manager = SequenceManager(
            [
                ManagerDecision(
                    action=ManagerAction.UPDATE_STATE,
                    state_update=StateUpdate(),
                ),
                ManagerDecision(action=ManagerAction.RESUME_WORKER),
                ManagerDecision(action=ManagerAction.ABSTAIN),
            ]
        )
        workspace = CountingWorkspace()

        result = StateGuardHarness(
            worker=worker,
            manager=manager,
            workspace=workspace,
        ).run(TaskSpec("preflight-retry", "Finish the task."))

        self.assertEqual(result.final_answer, "done")
        self.assertFalse(result.degraded)
        self.assertEqual(result.manager_actions, 3)
        self.assertEqual(workspace.restore_calls, 0)

    def test_exception_at_unit_end_releases_all_lifecycle_checkpoints(self):
        worker = WorkerAgent(
            ScriptedModelClient(
                [
                    json.dumps(
                        {
                            "type": "tool",
                            "reasoning": "continue",
                            "tool": "noop",
                            "arguments": {"value": 1},
                        }
                    )
                ]
            ),
            ToolRegistry(
                [FunctionTool("noop", "Return a value.", lambda value: value)]
            ),
        )
        harness = StateGuardHarness(
            worker=worker,
            manager=SequenceManager([ManagerDecision(action=ManagerAction.RESUME_WORKER)]),
            workspace=InMemoryWorkspace(),
            config=StateGuardConfig(max_worker_steps=1),
        )

        with self.assertRaises(RuntimeError):
            harness.run(TaskSpec("checkpoint-finally", "Keep working."))

        self.assertEqual(harness.checkpoints.retained_count, 0)

    def test_update_failure_restores_draft_and_trace_without_workspace(self):
        worker = WorkerAgent(
            ScriptedModelClient(
                [json.dumps({"type": "final", "answer": "done", "reasoning": "done"})]
            )
        )
        manager = SequenceManager(
            [
                ManagerDecision(
                    action=ManagerAction.OPEN_STATE,
                    state_header=StateHeader(
                        id="S1",
                        issue="Record the result",
                        constraints=(Constraint("Record the result."),),
                        relations=(StateRelation(StateRelationType.INIT),),
                    ),
                ),
                ManagerDecision(action=ManagerAction.RESUME_WORKER),
                ManagerDecision(
                    action=ManagerAction.UPDATE_STATE,
                    state_update=StateUpdate(
                        conclusions=(Conclusion("The worker finished."),),
                        traced_step_ids=(1,),
                    ),
                ),
            ]
        )
        workspace = CountingWorkspace()
        trace = CountingTraceBuffer()
        runtime = StateGuardRuntime.create(
            worker=worker,
            manager=manager,
            workspace=workspace,
            draft_store=FailingUpdateDraftStore(),
            trace_buffer=trace,
        )

        result = StateGuardHarness(runtime=runtime).run(
            TaskSpec("update-failure", "Finish and record the result.")
        )

        self.assertEqual(result.final_answer, "done")
        self.assertTrue(result.degraded)
        self.assertEqual(trace.restore_calls, 1)
        self.assertEqual(workspace.restore_calls, 0)

    def test_commit_failure_restores_control_plane_without_workspace(self):
        worker = WorkerAgent(
            ScriptedModelClient(
                [json.dumps({"type": "final", "answer": "done", "reasoning": "done"})]
            )
        )
        manager = SequenceManager(
            [
                ManagerDecision(
                    action=ManagerAction.OPEN_STATE,
                    state_header=StateHeader(
                        id="S1",
                        issue="Record the result",
                        constraints=(Constraint("Record the result."),),
                        relations=(StateRelation(StateRelationType.INIT),),
                    ),
                ),
                ManagerDecision(action=ManagerAction.RESUME_WORKER),
                ManagerDecision(
                    action=ManagerAction.UPDATE_STATE,
                    state_update=StateUpdate(
                        conclusions=(Conclusion("The worker finished."),),
                        traced_step_ids=(1,),
                    ),
                ),
                ManagerDecision(
                    action=ManagerAction.FINALIZE_RELATIONS,
                    relation_finalization=RelationFinalization(
                        mode=RelationFinalizationMode.CONFIRM,
                        relations=(StateRelation(StateRelationType.INIT),),
                        reason="This is the initial checked state.",
                    ),
                ),
                ManagerDecision(action=ManagerAction.COMMIT_STATE),
            ]
        )
        workspace = CountingWorkspace()
        store = StateStore()
        graph = FailingGraph()
        runtime = StateGuardRuntime.create(
            worker=worker,
            manager=manager,
            workspace=workspace,
            state_store=store,
            graph=graph,
        )

        result = StateGuardHarness(runtime=runtime).run(
            TaskSpec("commit-failure", "Finish and record the result.")
        )

        self.assertEqual(result.final_answer, "done")
        self.assertTrue(result.degraded)
        self.assertEqual(len(store), 0)
        self.assertEqual(graph.nodes, {})
        self.assertEqual(graph.edges, [])
        self.assertEqual(workspace.restore_calls, 0)

    def test_runtime_rejects_mismatched_checkpoint_components(self):
        worker = WorkerAgent(ScriptedModelClient([]))
        workspace = InMemoryWorkspace()
        incomplete = CheckpointManager({"worker": worker, "workspace": workspace})
        with self.assertRaises(ValueError):
            StateGuardRuntime.create(
                worker=worker,
                manager=None,
                workspace=workspace,
                checkpoints=incomplete,
            )

    def test_runtime_rejects_executor_bound_to_another_workspace(self):
        worker = WorkerAgent(ScriptedModelClient([]))
        workspace = InMemoryWorkspace()
        wrong_executor = TrustedPythonExecutor(InMemoryWorkspace())
        with self.assertRaises(ValueError):
            StateGuardRuntime.create(
                worker=worker,
                manager=None,
                workspace=workspace,
                worker_executor=wrong_executor,
            )

    def test_rejected_branch_never_commits(self):
        model = ScriptedModelClient(
            [
                json.dumps({"type": "final", "answer": "41", "reasoning": "mental arithmetic"}),
                json.dumps({"type": "final", "answer": "42", "reasoning": "rechecked"}),
            ]
        )
        worker = WorkerAgent(model)
        manager = SnapshotAwareRepairThenCommitManager()
        harness = StateGuardHarness(
            worker=worker,
            manager=manager,
            workspace=InMemoryWorkspace(),
            config=StateGuardConfig(max_repairs=3),
        )
        result = harness.run(TaskSpec("arithmetic", "Return the value of 6 * 7."))

        self.assertEqual(result.final_answer, "42")
        self.assertEqual(result.repair_count, 1)
        self.assertEqual(len(result.committed_states), 1)
        self.assertEqual(result.committed_states[0].used_variables[0].value, 42)
        self.assertNotIn("41", str(result.committed_states[0].to_dict()))
        self.assertEqual(manager.restore_calls, 0)
        self.assertEqual(
            [record["status"] for record in harness.trace_buffer.history()],
            ["rejected", "accepted"],
        )
        self.assertEqual(harness.checkpoints.retained_count, 0)

    def test_trace_selection_is_contiguous_before_repair(self):
        trace = TraceBuffer()
        trace.start_unit("normal-state")
        for step_id in (1, 2, 3):
            trace.append(
                ReActStep(
                    step_id=step_id,
                    action=AgentAction(
                        kind="final",
                        reasoning=f"step {step_id}",
                        answer=str(step_id),
                    ),
                    observation=None,
                    done=step_id == 3,
                )
            )

        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            trace.consume((1, 3), allow_sparse=False)

        self.assertEqual(
            [record["status"] for record in trace.history()],
            ["pending", "pending", "pending"],
        )

    def test_post_repair_state_can_commit_ordered_sparse_trace_subset(self):
        model = ScriptedModelClient(
            [
                json.dumps(
                    {"type": "final", "answer": "unsupported", "reasoning": "guess"}
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "reasoning": "produce supporting evidence",
                        "tool": "noop",
                        "arguments": {"value": 2},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "reasoning": "irrelevant intermediate action",
                        "tool": "noop",
                        "arguments": {"value": 3},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool",
                        "reasoning": "verify the corrected result",
                        "tool": "noop",
                        "arguments": {"value": 4},
                    }
                ),
                json.dumps(
                    {"type": "final", "answer": "supported", "reasoning": "verified"}
                ),
            ]
        )
        worker = WorkerAgent(
            model,
            ToolRegistry(
                [FunctionTool("noop", "Return one test value.", lambda value: value)]
            ),
        )
        harness = StateGuardHarness(
            worker=worker,
            manager=SparseRepairThenCommitManager(),
            workspace=InMemoryWorkspace(),
        )

        result = harness.run(
            TaskSpec("sparse-repair-trace", "Produce a supported rewritten answer.")
        )

        self.assertEqual(result.final_answer, "supported")
        self.assertEqual(result.repair_count, 1)
        self.assertEqual(len(result.committed_states), 1)
        self.assertEqual(result.committed_states[0].source_step_start, 2)
        self.assertEqual(result.committed_states[0].source_step_end, 5)
        self.assertEqual(
            [record["status"] for record in harness.trace_buffer.history()],
            ["rejected", "accepted", "passed", "accepted", "accepted"],
        )
        self.assertFalse(result.degraded)

    def test_manager_none_is_an_uninterrupted_worker_baseline(self):
        responses = [
            json.dumps(
                {
                    "type": "tool",
                    "reasoning": "execute the calculation",
                    "tool": "multiply",
                    "arguments": {"left": 6, "right": 7},
                }
            ),
            json.dumps({"type": "final", "answer": "42", "reasoning": "used tool output"}),
        ]
        tools = ToolRegistry(
            [FunctionTool("multiply", "Multiply two numbers.", lambda left, right: left * right)]
        )
        direct_model = RecordingModelClient(responses)
        direct_worker = WorkerAgent(direct_model, tools)
        direct_worker.start("Return 6 * 7.")
        direct_answer = direct_worker.run()

        baseline_model = RecordingModelClient(responses)
        result = StateGuardHarness(
            worker=WorkerAgent(baseline_model, tools),
            manager=None,
            workspace=InMemoryWorkspace(),
        ).run(TaskSpec("baseline", "Return 6 * 7."))

        self.assertEqual(result.final_answer, direct_answer)
        self.assertEqual(result.worker_steps, 2)
        self.assertEqual(result.manager_actions, 0)
        self.assertEqual(result.repair_count, 0)
        self.assertEqual(result.committed_states, ())
        self.assertEqual(baseline_model.message_snapshots, direct_model.message_snapshots)

    def test_manager_none_does_not_apply_manager_gt_firewall(self):
        worker = WorkerAgent(
            ScriptedModelClient(
                [json.dumps({"type": "final", "answer": "worker-only", "reasoning": "done"})]
            )
        )
        result = StateGuardHarness(
            worker=worker,
            manager=None,
            workspace=InMemoryWorkspace(),
        ).run(
            TaskSpec(
                "baseline-with-eval-metadata",
                "Answer without manager assistance.",
                metadata={"gold_answer": "kept only by the evaluator"},
            )
        )
        self.assertEqual(result.final_answer, "worker-only")

    def test_turn_baseline_preserves_prior_context_across_queries(self):
        model = RecordingModelClient(
            [
                json.dumps({"type": "final", "answer": "first", "reasoning": "turn one"}),
                json.dumps({"type": "final", "answer": "second", "reasoning": "turn two"}),
            ]
        )
        harness = StateGuardHarness(
            worker=WorkerAgent(model),
            manager=None,
            workspace=InMemoryWorkspace(),
        )
        first = harness.run(TaskSpec("turn-1", "Do the first analysis."))
        second = harness.run(TaskSpec("turn-2", "Now use it for the second analysis."))

        self.assertEqual((first.final_answer, second.final_answer), ("first", "second"))
        second_prompt = model.message_snapshots[1]
        self.assertTrue(any(message.content == "Do the first analysis." for message in second_prompt))
        self.assertTrue(any(message.content == "Now use it for the second analysis." for message in second_prompt))
        self.assertTrue(any('"answer": "first"' in message.content for message in second_prompt))

    def test_low_confidence_repair_is_rejected(self):
        with self.assertRaises(ValueError):
            ManagerDecision(
                action=ManagerAction.REPAIR,
                confidence=0.4,
                evidence=AnalyticalEvidence(0.4, ("maybe",), ("weak",)),
                error_hint=ErrorHint(ERROR_HINT_PROMPT, ("x",), "Might be wrong."),
            )

    def test_longds_reselection_requires_high_confidence_conflict(self):
        with self.assertRaises(ValueError):
            ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                confidence=0.7,
                relation_finalization=RelationFinalization(
                    mode=RelationFinalizationMode.RESELECT,
                    relations=(StateRelation(StateRelationType.BRANCH, "S2"),),
                    reason="Select again from current state and the compact index.",
                    conflict_evidence=(
                        "The current state uses S2 output but the provisional state was S1.",
                    ),
                ),
            )

    def test_heavy_repair_preserves_context_and_removes_only_variables(self):
        worker = WorkerAgent(ScriptedModelClient([]))
        worker.start("test")
        workspace = InMemoryWorkspace(variables={"clean": 1})
        checkpoints = CheckpointManager({"worker": worker, "workspace": workspace})
        clean = checkpoints.capture("clean")
        workspace.variables["dirty"] = 999
        workspace.artifacts["bad.csv"] = "polluted"
        worker.inject_observation("polluted branch")
        original = checkpoints.capture("original")
        decision = ManagerDecision(
            action=ManagerAction.REPAIR,
            confidence=0.99,
            evidence=AnalyticalEvidence(
                0.99,
                ("Only validated variables may be reused.",),
                ("dirty was created on the rejected branch",),
            ),
            error_hint=ErrorHint(ERROR_HINT_PROMPT, ("dirty",), "dirty came from the rejected branch."),
            cleanup=CleanupPlan(remove_variables=("dirty",)),
        )
        session = RepairSession(clean)
        session.begin_state("S1", clean)
        session.attempts = 2
        directive = RepairController().apply(
            decision=decision,
            session=session,
            current_branch=original,
            worker=worker,
            workspace=workspace,
        )
        self.assertEqual(directive, RepairDirective.RETRY)
        self.assertEqual(workspace.variables, {"clean": 1})
        self.assertIn("bad.csv", workspace.artifacts)
        self.assertTrue(any(message.content == "polluted branch" for message in worker.messages))
        self.assertIn("<error_hint>", worker.messages[-1].content)

    def test_repair_controller_uses_two_light_then_one_heavy(self):
        worker = WorkerAgent(ScriptedModelClient([]))
        worker.start("test")
        workspace = InMemoryWorkspace(variables={"keep_for_light": 1})
        checkpoints = CheckpointManager({"worker": worker, "workspace": workspace})
        interval_start = checkpoints.capture("interval")
        current = checkpoints.capture("current")
        decision = ManagerDecision(
            action=ManagerAction.REPAIR,
            confidence=0.99,
            evidence=AnalyticalEvidence(
                0.99,
                ("The calculation must use executed evidence.",),
                ("The current result has no execution observation.",),
            ),
            error_hint=ErrorHint(
                ERROR_HINT_PROMPT,
                ("result",),
                "The result is unsupported by an execution observation.",
            ),
            cleanup=CleanupPlan(remove_variables=("keep_for_light",)),
        )
        session = RepairSession(interval_start)
        session.begin_state("S1", interval_start)
        controller = RepairController(max_repairs=3, light_repair_attempts=2)
        directives = []
        for _ in range(3):
            directives.append(
                controller.apply(
                    decision=decision,
                    session=session,
                    current_branch=current,
                    worker=worker,
                    workspace=workspace,
                )
            )

        self.assertEqual(directives, [RepairDirective.RETRY] * 3)
        self.assertEqual([record["mode"] for record in session.records], ["light", "light", "heavy"])
        self.assertNotIn("keep_for_light", workspace.variables)
        exhausted = controller.abandon_state(session, checkpoints)
        self.assertEqual(exhausted, RepairDirective.RESTORED_ORIGINAL)
        self.assertIn("keep_for_light", workspace.variables)

    def test_each_state_receives_an_independent_repair_budget(self):
        worker = WorkerAgent(ScriptedModelClient([]))
        worker.start("test")
        workspace = InMemoryWorkspace()
        checkpoints = CheckpointManager({"worker": worker, "workspace": workspace})
        start = checkpoints.capture("start")
        current = checkpoints.capture("current")
        decision = ManagerDecision(
            action=ManagerAction.REPAIR,
            confidence=0.99,
            evidence=AnalyticalEvidence(
                0.99,
                ("The current state must use executed evidence.",),
                ("The current state has no execution result.",),
            ),
            error_hint=ErrorHint(
                ERROR_HINT_PROMPT,
                ("result",),
                "The current state result is unsupported.",
            ),
        )
        session = RepairSession(start)
        controller = RepairController()
        session.begin_state("S1", start)
        for _ in range(3):
            controller.apply(
                decision=decision,
                session=session,
                current_branch=current,
                worker=worker,
                workspace=workspace,
            )
        self.assertEqual(session.attempts, 3)

        next_state_start = checkpoints.capture("S2:start")
        session.begin_state("S2", next_state_start)
        self.assertEqual(session.state_id, "S2")
        self.assertEqual(session.attempts, 0)
        self.assertIsNone(session.original_branch)

    def test_manager_writes_exact_relations_before_trace_and_harness_only_executes(self):
        predecessor = AnalyticalState(
            id="S1",
            issue="Compute the value",
            constraints=(Constraint("Compute 6 * 7."),),
            used_variables=(VariableRef("result", "S1", value=42),),
            conclusions=(Conclusion("The value is 42."),),
            relations=(StateRelation(StateRelationType.INIT),),
        )
        store = StateStore()
        store.commit(predecessor)
        graph = StateRelationGraph()
        graph.add_state(predecessor)
        model = RecordingModelClient(
            [json.dumps({"type": "final", "answer": "42", "reasoning": "used S1"})]
        )
        manager = RelationFirstManager()
        result = StateGuardHarness(
            worker=WorkerAgent(model),
            manager=manager,
            workspace=InMemoryWorkspace(),
            state_store=store,
            graph=graph,
        ).run(TaskSpec("relation", "Report the previously computed value."))

        self.assertIsNone(manager.observations[0].worker_step)
        self.assertEqual(manager.observations[0].untraced_steps, ())
        self.assertFalse(hasattr(manager.observations[0], "stored_states"))
        self.assertFalse(hasattr(manager.observations[1], "relation_states"))
        self.assertEqual(store.load_state_json("S1")["used_variables"][0]["value"], 42)
        first_worker_prompt = "\n".join(message.content for message in model.message_snapshots[0])
        self.assertIn("<analytical_state_hint>", first_worker_prompt)
        hint_json = first_worker_prompt.split("<analytical_state_hint>\n", 1)[1].split(
            "\n</analytical_state_hint>", 1
        )[0]
        hint_states = json.loads(hint_json)
        self.assertEqual(len(hint_states), 1)
        self.assertEqual(
            set(hint_states[0]),
            {"id", "issue", "conclusions", "relations"},
        )
        self.assertEqual(hint_states[0]["id"], "S1")
        self.assertNotIn("used_variables", hint_states[0])
        self.assertEqual(graph.ancestors("S2"), ("S1",))
        self.assertEqual([state.id for state in result.committed_states], ["S1", "S2"])
        self.assertTrue(result.committed_states[-1].metadata["relations_finalized"])
        self.assertEqual(
            result.committed_states[-1].metadata["relation_finalization_mode"],
            "confirm",
        )

    def test_longds_clear_conflict_reselects_using_current_state_and_compact_index(self):
        s1 = AnalyticalState(
            id="S1",
            issue="Produce the first result",
            constraints=(Constraint("Use source A."),),
            used_variables=(VariableRef("result_a", "S1", value=10),),
            conclusions=(Conclusion("Source A result is 10."),),
            relations=(StateRelation(StateRelationType.INIT),),
        )
        s2 = AnalyticalState(
            id="S2",
            issue="Produce an alternative result",
            constraints=(Constraint("Use source B."),),
            used_variables=(VariableRef("result_b", "S2", value=20),),
            conclusions=(Conclusion("Source B result is 20."),),
            relations=(StateRelation(StateRelationType.BRANCH, "S1"),),
        )
        store = StateStore()
        graph = StateRelationGraph()
        for state in (s1, s2):
            store.commit(state)
            graph.add_state(state)
        model = RecordingModelClient(
            [json.dumps({"type": "final", "answer": "used S2", "reasoning": "actual trace"})]
        )
        manager = RelationReselectManager()
        result = StateGuardHarness(
            worker=WorkerAgent(model),
            manager=manager,
            workspace=InMemoryWorkspace(),
            state_store=store,
            graph=graph,
        ).run(TaskSpec("relation-conflict", "Continue the relevant previous result."))

        state = result.committed_states[-1]
        reselection_view = manager.observations[3]
        self.assertEqual(
            reselection_view.current_draft["conclusions"][0],
            "The worker actually reused S2.",
        )
        self.assertFalse(hasattr(reselection_view, "stored_states"))
        self.assertFalse(hasattr(reselection_view, "relation_states"))
        self.assertEqual(
            {item["id"] for item in store.load_store_json()},
            {"S1", "S2", "S3"},
        )
        self.assertEqual(state.metadata["provisional_relations"][0]["related_state_id"], "S1")
        self.assertEqual(state.metadata["final_relations"][0]["related_state_id"], "S2")
        self.assertEqual(state.metadata["relation_finalization_mode"], "reselect")
        self.assertTrue(state.metadata["relation_conflict_evidence"])
        self.assertEqual(graph.edges[-1]["source"], "S2")
        self.assertNotEqual(graph.edges[-1]["source"], "S1")

    def test_manager_rechecks_between_two_light_one_heavy_then_abandon_state(self):
        model = ScriptedModelClient(
            [
                json.dumps(
                    {"type": "final", "answer": f"unsupported-{index}", "reasoning": "guess"}
                )
                for index in range(4)
            ]
        )
        manager = FixedRepairSequenceManager()
        harness = StateGuardHarness(
            worker=WorkerAgent(model),
            manager=manager,
            workspace=InMemoryWorkspace(),
        )
        result = harness.run(TaskSpec("fixed-repair", "Return a verified answer."))

        checked_outputs = [
            observation.worker_step.action.answer
            for observation in manager.observations[2:6]
        ]
        self.assertEqual(
            checked_outputs,
            ["unsupported-0", "unsupported-1", "unsupported-2", "unsupported-3"],
        )
        self.assertEqual(
            [observation.repair_attempts for observation in manager.observations[2:6]],
            [0, 1, 2, 3],
        )
        self.assertEqual(result.repair_count, 3)
        self.assertEqual(result.abstained_intervals, 1)
        self.assertEqual(result.committed_states, ())
        self.assertEqual(harness._next_state_number, 2)
        self.assertEqual(manager.observations[-1].available_state_id, "S2")
        self.assertEqual(result.final_answer, "unsupported-0")
        self.assertEqual(harness.checkpoints.retained_count, 0)

    def test_fixed_step_adapter_reviews_five_steps_then_manager_forms_state(self):
        responses = [
            json.dumps(
                {
                    "type": "tool",
                    "reasoning": "continue the segment",
                    "tool": "noop",
                    "arguments": {"value": index},
                }
            )
            for index in range(1, 5)
        ]
        responses.append(
            json.dumps({"type": "final", "answer": "complete", "reasoning": "step five"})
        )
        model = RecordingModelClient(responses)
        worker = WorkerAgent(
            model,
            ToolRegistry([FunctionTool("noop", "Record one test step.", lambda value: {"value": value})]),
        )
        manager = FixedWindowManager()
        result = StateGuardHarness(
            worker=worker,
            manager=manager,
            workspace=InMemoryWorkspace(),
            flow_adapter=FixedStepFlowAdapter(window_size=5),
        ).run(TaskSpec("window", "Complete a five-step analysis."))

        self.assertEqual(len(manager.observations), 5)
        self.assertEqual(manager.observations[1].event_type, "STEP_WINDOW")
        self.assertEqual(len(manager.observations[1].untraced_steps), 5)
        self.assertEqual(
            manager.observations[1].flow_policy["relation_timing"],
            "segment_complete",
        )
        self.assertNotIn(
            "<analytical_state_hint>",
            "\n".join(message.content for snapshot in model.message_snapshots for message in snapshot),
        )
        relation_selection_view = manager.observations[3]
        self.assertEqual(relation_selection_view.current_draft["relations"], [])
        self.assertEqual(
            relation_selection_view.current_draft["conclusions"][0],
            "The five-step segment completed.",
        )
        self.assertEqual(result.committed_states[-1].source_step_end, 5)
        self.assertTrue(result.committed_states[-1].metadata["relations_finalized"])
        self.assertEqual(
            result.committed_states[-1].metadata["relation_finalization_mode"],
            "select",
        )
        self.assertEqual(result.committed_states[-1].metadata["provisional_relations"], [])

    def test_fixed_step_manager_selects_arbitrary_prefix_and_keeps_trailing_steps(self):
        responses = [
            json.dumps(
                {
                    "type": "tool",
                    "reasoning": "continue the analysis",
                    "tool": "noop",
                    "arguments": {"value": index},
                }
            )
            for index in range(1, 6)
        ]
        responses.append(
            json.dumps({"type": "final", "answer": "done", "reasoning": "step six"})
        )
        worker = WorkerAgent(
            ScriptedModelClient(responses),
            ToolRegistry([FunctionTool("noop", "Record one test step.", lambda value: {"value": value})]),
        )
        manager = ArbitraryBoundaryManager()

        result = StateGuardHarness(
            worker=worker,
            manager=manager,
            workspace=InMemoryWorkspace(),
            flow_adapter=FixedStepFlowAdapter(window_size=5),
        ).run(TaskSpec("arbitrary-window", "Complete a multi-stage analysis."))

        state = result.committed_states[-1]
        self.assertEqual((state.source_step_start, state.source_step_end), (1, 3))
        self.assertEqual(
            (
                manager.observations[1].candidate_start_step,
                manager.observations[1].candidate_end_step,
            ),
            (1, 5),
        )
        self.assertEqual(
            (
                manager.observations[5].candidate_start_step,
                manager.observations[5].candidate_end_step,
            ),
            (4, 5),
        )
        self.assertEqual(
            (
                manager.observations[-1].candidate_start_step,
                manager.observations[-1].candidate_end_step,
            ),
            (4, 6),
        )
        self.assertEqual(result.final_answer, "done")


if __name__ == "__main__":
    unittest.main()
