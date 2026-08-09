from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from stateguard.adapters.dacomp.adapter import DACompAdapter
from stateguard.adapters.dacomp.dataset import DACompDataset, DACompTask, DACompTrack
from stateguard.adapters.dacomp.worker_da import DACompDAWorkerAgent
from stateguard.adapters.dacomp.workspace import DACompWorkspace
from stateguard.adapters.dacomp.worker_de import (
    DACompDEWorkerAgent,
    NativeCodeActStep,
)
from stateguard.adapters.dacomp.workflow import DACompWorkflow
from stateguard.core.events import ReActStep
from stateguard.core.models import AgentAction, Message, ToolResult
from stateguard.state.draft import (
    RelationFinalization,
    RelationFinalizationMode,
    StateDraft,
    StateHeader,
)
from stateguard.state.models import Constraint, StateRelation, StateRelationType


class FakeWorker:
    def __init__(self, track: str = "de"):
        self.track = track
        self._messages = []
        self._done = False
        self._answer = None
        self.accepted_steps = 0
        self.closed = False

    @property
    def messages(self):
        return tuple(self._messages)

    @property
    def done(self):
        return self._done

    @property
    def final_answer(self):
        return self._answer

    def start(self, prompt, system_prompt=None):
        assert system_prompt is None
        self._messages.append(Message("user", str(prompt)))

    def continue_turn(self, prompt):
        raise AssertionError(prompt)

    def step(self):
        self.accepted_steps += 1
        self._done = True
        self._answer = "finished"
        return ReActStep(
            self.accepted_steps,
            AgentAction(kind="final", answer="finished"),
            None,
            True,
            raw_model_output="finish",
        )

    def inject_observation(self, content, metadata=None):
        self._messages.append(Message("user", str(content), metadata=metadata or {}))
        self._done = False

    def snapshot(self):
        return copy.deepcopy((self._messages, self._done, self._answer, self.accepted_steps))

    def restore(self, snapshot):
        self._messages, self._done, self._answer, self.accepted_steps = copy.deepcopy(snapshot)

    def trajectory(self):
        return {} if self.track == "da" else []

    def close(self):
        self.closed = True


class FakeDAAction:
    def __init__(self, code="ls"):
        self.code = code

    def __str__(self):
        return f'Bash(code="{self.code}")'


class FakeDAAgent:
    def __init__(self):
        self.thoughts = []
        self.responses = []
        self.actions = []
        self.observations = []
        self.history_messages = []
        self.codes = []
        self._last_repetition_signature = None

    def set_env_and_task(self, environment):
        self.environment = environment

    def predict(self, observation):
        action = FakeDAAction()
        response = 'Thought: inspect\nAction: Bash(code="ls")'
        self.thoughts.append("inspect")
        self.responses.append(response)
        self.actions.append(action)
        self.observations.append(observation)
        return response, action

    def get_trajectory(self):
        return {"trajectory": []}


class FakeInvalidDAAgent(FakeDAAgent):
    def predict(self, observation):
        del observation
        response = "invalid response"
        self.responses.append(response)
        return response, None


class FakeDAEnvironment:
    def step(self, action):
        del action
        return "ok", False

    def post_process(self):
        return {}

    def close(self):
        return None


class FakeCodeActSession:
    def __init__(self):
        self.started = False
        self.user_messages = []
        self.index = 0

    def start(self, instruction):
        self.started = True
        self.user_messages.append(instruction)

    def advance(self):
        self.index += 1
        if self.index == 1:
            return NativeCodeActStep(
                "execute_bash", "inspect", {"command": "ls"}, "ok", True, True,
                raw_action="CmdRunAction(command='ls')",
            )
        return NativeCodeActStep(
            "finish", "done", {}, "", False, True, True, "complete", "finish", ""
        )

    def inject_user_message(self, content):
        self.user_messages.append(content)

    def snapshot(self):
        return copy.deepcopy((self.started, self.user_messages, self.index))

    def restore(self, snapshot):
        self.started, self.user_messages, self.index = copy.deepcopy(snapshot)

    def messages(self):
        return tuple(Message("user", item) for item in self.user_messages)

    def trajectory(self):
        return []

    def close(self):
        return None


class DACompAdapterTests(unittest.TestCase):
    def test_dataset_keeps_eval_material_out_of_task_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            da_tasks = root / "dacomp-da" / "tasks"
            task_dir = da_tasks / "dacomp-001"
            task_dir.mkdir(parents=True)
            (task_dir / "dacomp-001.sqlite").write_bytes(b"sqlite")
            (da_tasks / "dacomp-da.jsonl").write_text(
                json.dumps({"instance_id": "dacomp-001", "instruction": "analyze"}) + "\n",
                encoding="utf-8",
            )
            (root / "dacomp-de" / "tasks").mkdir(parents=True)
            task = DACompDataset(root).load(DACompTrack.DA_STAGE1)[0]
            self.assertEqual(task.instruction, "analyze")
            self.assertNotIn("rubric", json.dumps(task.task_spec().metadata).lower())
            self.assertNotIn("gold", json.dumps(task.task_spec().metadata).lower())

    def test_workflow_is_pause_cadence_not_boundary_and_has_no_hint(self):
        workflow = DACompWorkflow(5)
        lifecycle = workflow.lifecycle_prompt()
        self.assertIn("Pause after every 5 accepted native Worker actions", lifecycle)
        self.assertIn("STATE-FORMATION DECISION", lifecycle)
        self.assertIn("A plan, navigation step, repeated inspection", lifecycle)
        self.assertIn("uncertain, RESUME_WORKER", lifecycle)
        self.assertNotIn("{{REVIEW_CADENCE}}", lifecycle)
        step = ReActStep(
            5,
            AgentAction(kind="tool", tool_name="x"),
            ToolResult("x", True, "ok"),
            False,
        )
        self.assertFalse(workflow.should_review(step, 4))
        self.assertTrue(workflow.should_review(step, 5))
        self.assertEqual(workflow.hint_state_ids(("S1",)), ())
        with self.assertRaises(ValueError):
            workflow.validate_state_open(
                StateHeader(
                    id="S2",
                    constraints=(Constraint("query"),),
                    relations=(StateRelation(StateRelationType.PROGRESS, "S1"),),
                ),
                (step,),
            )

    def test_da_budget_exhaustion_stops_facade_but_is_not_official_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            task = DACompTask(
                "dacomp-test",
                DACompTrack.DA_STAGE1,
                "analyze",
                source,
                {"official_record": {"instruction": "analyze"}},
            )
            worker = DACompDAWorkerAgent(
                task=task,
                workspace=DACompWorkspace(root / "workspace", source),
                official_root=root,
                model="fake",
                max_steps=1,
                agent=FakeDAAgent(),
                environment=FakeDAEnvironment(),
            )
            worker.start("analyze")
            step = worker.step()
            self.assertTrue(step.done)
            self.assertTrue(worker.done)
            self.assertFalse(worker.official_finished)
            self.assertEqual(worker.final_answer, "")

    def test_da_parse_retry_exhaustion_matches_official_nonfinished_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            task = DACompTask(
                "dacomp-test",
                DACompTrack.DA_STAGE1,
                "analyze",
                source,
                {"official_record": {"instruction": "analyze"}},
            )
            worker = DACompDAWorkerAgent(
                task=task,
                workspace=DACompWorkspace(root / "workspace", source),
                official_root=root,
                model="fake",
                agent=FakeInvalidDAAgent(),
                environment=FakeDAEnvironment(),
            )
            worker.start("analyze")
            step = worker.step()
            self.assertTrue(step.done)
            self.assertEqual(step.metadata["official_stop_reason"], "parse_retry_exhausted")
            self.assertFalse(worker.official_finished)
            self.assertEqual(worker.accepted_steps, 0)

    def test_de_codeact_facade_counts_native_actions(self):
        session = FakeCodeActSession()
        worker = DACompDEWorkerAgent(session, max_steps=200)
        worker.start("task")
        first = worker.step()
        self.assertEqual(first.action.tool_name, "execute_bash")
        self.assertTrue(first.observation.data["execution_attempted"])
        self.assertFalse(worker.done)
        second = worker.step()
        self.assertTrue(second.done)
        self.assertEqual(worker.final_answer, "complete")
        self.assertEqual(worker.accepted_steps, 2)

    def test_adapter_manager_none_uses_native_worker_and_official_de_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "DAComp-main"
            (root / "dacomp-da" / "tasks").mkdir(parents=True)
            (root / "dacomp-da" / "tasks" / "dacomp-da.jsonl").write_text("", encoding="utf-8")
            task_dir = root / "dacomp-de" / "tasks" / "dacomp-de-evol-001"
            (task_dir / "config").mkdir(parents=True)
            (task_dir / "sql").mkdir()
            (task_dir / "config" / "layer_dependencies.yaml").write_text("layers: {}\n")
            (task_dir / "question.md").write_text("Add a metric.\n")
            (task_dir / "run.py").write_text("print('ok')\n")
            workers = []

            def factory(task, workspace):
                del task, workspace
                worker = FakeWorker("de")
                workers.append(worker)
                return worker

            adapter = DACompAdapter(
                dacomp_root=root,
                output_root=base / "results",
                track=DACompTrack.DE_EVOL,
                model="fake",
                worker_factory=factory,
            )
            task = adapter.load_tasks()[0]
            result = adapter.run_task(task, manager=None)
            self.assertIsNone(result.error)
            self.assertEqual(result.stateguard_result.manager_actions, 0)
            self.assertEqual(adapter.max_worker_steps, 30)
            self.assertTrue((result.run_dir / "result.json").is_file())
            self.assertTrue((result.run_dir / "workspace_summary.json").is_file())
            self.assertFalse((result.run_dir / "stateguard").exists())
            metadata = json.loads(
                (result.run_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            sidecar = Path(metadata["stateguard_artifact_dir"])
            self.assertEqual(
                sidecar,
                base
                / "results"
                / "_stateguard"
                / "de-evol"
                / "fake_default"
                / "dacomp-de-evol-001",
            )
            self.assertTrue((sidecar / "worker.jsonl").is_file())
            self.assertTrue((sidecar / "summary.json").is_file())
            official_result = json.loads(
                (result.run_dir / "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(official_result["summary"]),
                {"total_steps", "tool_calls", "bash_calls", "ipython_calls", "finish_calls"},
            )
            self.assertTrue(workers[0].closed)


if __name__ == "__main__":
    unittest.main()
