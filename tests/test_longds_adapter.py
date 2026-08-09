import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stateguard.adapters.longds.adapter import LongDSAdapter
from stateguard.adapters.longds import runner as longds_runner
from stateguard.adapters.longds.executor import LongDSProbeExecutor
from stateguard.adapters.longds.workflow import LongDSWorkflow
from stateguard.adapters.longds.worker import LongDSWorkerAgent
from stateguard.adapters.longds.workspace import LongDSWorkspace
from stateguard.state.draft import (
    RelationFinalization,
    RelationFinalizationMode,
    StateHeader,
    StateUpdate,
)
from stateguard.state.models import (
    Conclusion,
    Constraint,
    StateRelation,
    StateRelationType,
)
from stateguard.validation.models import ManagerAction, ManagerDecision


class FakeResponse:
    def __init__(self, value):
        self.value = value
        self.status_code = 200
        self.is_success = True

    def raise_for_status(self):
        return None

    def json(self):
        return self.value


class FakeClient:
    def __init__(self):
        self.namespace = {}
        self.restarts = 0

    def post(self, url, json=None):
        if url.endswith("/restart"):
            self.namespace = {}
            self.restarts += 1
            return FakeResponse({"status": "ok"})
        code = (json or {})["code"]
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(code, "<fake-longds>", "exec"), self.namespace, self.namespace)
            outputs = []
            if stdout.getvalue():
                outputs.append({"type": "stream", "text": stdout.getvalue()})
        except Exception as exc:
            outputs = [
                {"type": "error", "name": type(exc).__name__, "value": str(exc)}
            ]
        return FakeResponse({"outputs": outputs})

    def get(self, url):
        del url
        return FakeResponse({"ready": True})


class FakeToolGroup:
    def __init__(self):
        self.manager_url = "http://fake"
        self.allocated_container = None
        self.client = FakeClient()

    def allocate_container(self):
        self.allocated_container = 1
        return 1

    def deallocate_container(self):
        self.allocated_container = None

    def get_tool_names(self):
        return ["python"]


def clean_output(outputs):
    rendered = []
    for output in outputs:
        if output.get("type") == "stream":
            rendered.append(output.get("text", ""))
        elif output.get("type") == "error":
            rendered.append(f"{output.get('name')}: {output.get('value')}")
    return "".join(rendered)


class FakeEnvironment:
    def __init__(self, max_turns=40):
        self.max_turns = max_turns
        self.turns = 0
        self.chat_history = []
        self.tool_group = FakeToolGroup()
        self.closed = False

    def init(self, prompt, **extras):
        del extras
        self.turns = 0
        self.chat_history = [dict(message) for message in prompt]
        self.tool_group.allocate_container()
        return [dict(message) for message in prompt], {}

    def reset_turns(self):
        self.turns = 0

    def step(self, raw):
        self.turns += 1
        if "</python>" in raw:
            action = raw.split("</python>", 1)[0] + "</python>"
        elif "</answer>" in raw:
            action = raw.split("</answer>", 1)[0] + "</answer>"
        else:
            action = raw
        answer = ""
        if "<answer>" in action and "</answer>" in action:
            answer = action.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        done = self.turns >= self.max_turns or bool(answer)
        if done:
            return {
                "postprocessed_action": action,
                "observations": [],
                "done": True,
                "metadata": {"final_answer": answer, "code_executed": False},
            }
        observations = []
        if "<python>" in action and "</python>" in action:
            code = action.split("<python>", 1)[1].split("</python>", 1)[0]
            output = self.tool_group.execute_code(code)
            observations = [
                {"role": "user", "content": f"\n<information>{output}</information>\n"}
            ]
        else:
            observations = [
                {"role": "user", "content": "<information>No python code found.</information>"}
            ]
        return {
            "postprocessed_action": action,
            "observations": observations,
            "done": False,
            "metadata": {"final_answer": ""},
        }

    def close(self):
        self.closed = True
        self.tool_group.deallocate_container()


class FailingSecondInitEnvironment(FakeEnvironment):
    def __init__(self, max_turns=40):
        super().__init__(max_turns=max_turns)
        self.init_calls = 0

    def init(self, prompt, **extras):
        self.init_calls += 1
        if self.init_calls == 2:
            raise TimeoutError("simulated environment restart timeout")
        return super().init(prompt, **extras)


class FailedAllocationEnvironment(FakeEnvironment):
    def init(self, prompt, **extras):
        del extras
        self.turns = 0
        self.chat_history = [dict(message) for message in prompt]
        return [dict(message) for message in prompt], {
            # Reproduce current upstream DSGym's incorrect success metadata.
            "container_allocated": True
        }


class FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.snapshots = []

    def generate(self, messages):
        self.snapshots.append([dict(message) for message in messages])
        if not self.responses:
            raise RuntimeError("fake backend exhausted")
        return self.responses.pop(0)


class TwoTurnManager:
    def __init__(self):
        self.action_index = 0
        self.tasks = []

    def start_task(self, task):
        self.tasks.append(task)

    def act(self, observation):
        self.action_index += 1
        index = self.action_index
        if index == 1:
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    id="S1",
                    issue="Q1",
                    constraints=(Constraint("Answer Q1."),),
                    relations=(StateRelation(StateRelationType.INIT),),
                ),
            )
        if index == 2:
            return ManagerDecision(action=ManagerAction.RESUME_WORKER)
        if index == 3:
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    conclusions=(Conclusion("Q1 answer is 41."),),
                    traced_step_ids=tuple(step.step_id for step in observation.untraced_steps),
                ),
            )
        if index == 4:
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    RelationFinalizationMode.CONFIRM,
                    (StateRelation(StateRelationType.INIT),),
                    "First state remains initialization.",
                ),
            )
        if index == 5:
            return ManagerDecision(action=ManagerAction.COMMIT_STATE)
        if index == 6:
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    id="S2",
                    issue="Q2",
                    constraints=(Constraint("Answer Q2 using relevant checked work."),),
                    relations=(StateRelation(StateRelationType.PROGRESS, "S1"),),
                ),
            )
        if index == 7:
            return ManagerDecision(action=ManagerAction.RESUME_WORKER)
        if index == 8:
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    conclusions=(Conclusion("Q2 answer is 42."),),
                    traced_step_ids=tuple(step.step_id for step in observation.untraced_steps),
                ),
            )
        if index == 9:
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    RelationFinalizationMode.CONFIRM,
                    (StateRelation(StateRelationType.PROGRESS, "S1"),),
                    "The completed turn directly progresses from S1.",
                ),
            )
        return ManagerDecision(action=ManagerAction.COMMIT_STATE)


class LongDSAdapterTest(unittest.TestCase):
    def _write_dataset(self, root):
        task_root = root / "dataset" / "task" / "longds"
        raw_dir = task_root / "science" / "demo" / "task1"
        raw_dir.mkdir(parents=True)
        (task_root / "task_list.json").write_text(
            json.dumps(
                [
                    {
                        "task_domain": "science",
                        "dataset_name": "demo",
                        "task_id": "task1",
                    }
                ]
            ),
            encoding="utf-8",
        )
        (raw_dir / "task.json").write_text(
            json.dumps(
                [
                    {
                        "turn_id": 1,
                        "context": "C1",
                        "question": "Q1",
                        "answer": "41",
                        "extra_info": {"trajectory_id": 7},
                    },
                    {
                        "turn_id": 2,
                        "context": "C2",
                        "question": "Q2",
                        "answer": "42",
                    },
                ]
            ),
            encoding="utf-8",
        )
        return task_root

    def test_longds_uses_its_own_turn_lifecycle_prompt(self):
        workflow = LongDSWorkflow({}, "system")
        lifecycle = workflow.lifecycle_prompt()
        self.assertIn("LONGDS TURN LIFECYCLE", lifecycle)
        self.assertIn("do not inspect or interrupt inside the turn", lifecycle)
        self.assertIn("contiguous turn. After repair", lifecycle)
        self.assertNotIn("STATE-FORMATION DECISION", lifecycle)

    def test_baseline_preserves_official_messages_workspace_and_per_turn_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_root = self._write_dataset(root)
            backends = []
            environments = []

            def backend_factory():
                backend = FakeBackend(
                    [
                        "<reasoning>set x</reasoning><python>x = 41</python>",
                        "<answer>41</answer>",
                        "<reasoning>reuse x</reasoning><python>print(x + 1)</python>",
                        "<answer>42</answer>",
                    ]
                )
                backends.append(backend)
                return backend

            def environment_factory():
                environment = FakeEnvironment(max_turns=4)
                environments.append(environment)
                return environment

            adapter = LongDSAdapter(
                dsgym_root=root / "unused",
                dataset_root=task_root,
                output_root=root / "results",
                model="fake/model",
                max_steps_per_turn=4,
                backend_factory=backend_factory,
                environment_factory=environment_factory,
                clean_output=clean_output,
                system_prompt_template="OFFICIAL DATA={PATH}",
            )
            task = adapter.load_tasks()[0]
            self.assertNotIn("ground_truth", task.turns[0].public.task_spec().metadata)
            result = adapter.run_task(task, manager=None)

            self.assertEqual(len(backends), 1)
            self.assertEqual(len(environments), 1)
            conversation = result.trajectory["conversation"]
            self.assertEqual(sum(msg["role"] == "system" for msg in conversation), 1)
            expected_data = (
                root
                / "dataset"
                / "data"
                / "longds"
                / "science"
                / "demo"
                / "task1"
                / "data"
            )
            self.assertEqual(conversation[0]["content"], f"OFFICIAL DATA={expected_data}")
            self.assertEqual(conversation[1]["content"], "C1\nQuestion: Q1")
            self.assertTrue(any("42" in msg["content"] for msg in conversation))
            self.assertEqual([item["steps"] for item in result.turn_results], [2, 2])
            self.assertEqual([item["solution"] for item in result.turn_results], ["41", "42"])
            self.assertFalse(any("stateguard" in msg["content"].lower() for msg in conversation))
            for filename in ("traj.json", "results.json", "code.py"):
                self.assertTrue((result.run_dir / filename).exists())

    def test_managed_task_reuses_runtime_and_injects_related_state_as_information(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_root = self._write_dataset(root)
            backends = []
            environments = []

            def backend_factory():
                backend = FakeBackend(["<answer>41</answer>", "<answer>42</answer>"])
                backends.append(backend)
                return backend

            def environment_factory():
                environment = FakeEnvironment(max_turns=4)
                environments.append(environment)
                return environment

            adapter = LongDSAdapter(
                dsgym_root=root / "unused",
                dataset_root=task_root,
                output_root=root / "results",
                model="fake/model",
                max_steps_per_turn=4,
                backend_factory=backend_factory,
                environment_factory=environment_factory,
                clean_output=clean_output,
                system_prompt_template="OFFICIAL DATA={PATH}",
            )
            manager = TwoTurnManager()
            result = adapter.run_task(adapter.load_tasks()[0], manager=manager)

            self.assertEqual(len(backends), 1)
            # Probe allocation is lazy; no Manager probe action means only Worker env exists.
            self.assertEqual(len(environments), 1)
            self.assertEqual(len(manager.tasks), 2)
            self.assertNotIn("ground_truth", manager.tasks[0].metadata)
            self.assertEqual(
                result.stateguard_results[0].runtime_id,
                result.stateguard_results[1].runtime_id,
            )
            self.assertEqual(
                [state.id for state in result.stateguard_results[-1].committed_states],
                ["S1", "S2"],
            )
            second_snapshot = backends[0].snapshots[1]
            hint = [
                message["content"]
                for message in second_snapshot
                if "<analytical_state_hint>" in message["content"]
            ]
            self.assertEqual(len(hint), 1)
            self.assertTrue(hint[0].startswith("<information>"))
            self.assertIn('"id": "S1"', hint[0])
            self.assertNotIn("used_variables", hint[0])

    def test_task_exception_saves_completed_turns_as_official_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_root = self._write_dataset(root)
            environment = FailingSecondInitEnvironment(max_turns=4)
            adapter = LongDSAdapter(
                dsgym_root=root / "unused",
                dataset_root=task_root,
                output_root=root / "results",
                model="fake/model",
                max_steps_per_turn=4,
                reset_env_times=2,
                backend_factory=lambda: FakeBackend(["<answer>41</answer>"]),
                environment_factory=lambda: environment,
                clean_output=clean_output,
                system_prompt_template="OFFICIAL DATA={PATH}",
            )

            result = adapter.run_task(adapter.load_tasks()[0], manager=None)

            self.assertEqual(len(result.turn_results), 1)
            self.assertEqual(result.turn_results[0]["solution"], "41")
            self.assertFalse(result.trajectory["success"])
            self.assertEqual(
                result.trajectory["error"], "simulated environment restart timeout"
            )
            self.assertEqual(result.trajectory["metadata"]["error_type"], "TimeoutError")
            self.assertEqual(result.trajectory["total_turns"], 2)
            self.assertTrue(environment.closed)
            for filename in ("traj.json", "results.json", "code.py", "results_eval.json"):
                self.assertTrue((result.run_dir / filename).exists())
            saved_results = json.loads(
                (result.run_dir / "results.json").read_text(encoding="utf-8")
            )
            self.assertEqual([item["turn_id"] for item in saved_results], [1])

    def test_first_turn_init_failure_still_writes_empty_official_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_root = self._write_dataset(root)
            environment = FailingSecondInitEnvironment(max_turns=4)
            environment.init_calls = 1
            adapter = LongDSAdapter(
                dsgym_root=root / "unused",
                dataset_root=task_root,
                output_root=root / "results",
                model="fake/model",
                max_steps_per_turn=4,
                backend_factory=lambda: FakeBackend([]),
                environment_factory=lambda: environment,
                clean_output=clean_output,
                system_prompt_template="OFFICIAL DATA={PATH}",
            )

            result = adapter.run_task(adapter.load_tasks()[0], manager=None)

            self.assertEqual(result.turn_results, ())
            self.assertFalse(result.trajectory["success"])
            self.assertEqual(result.trajectory["metadata"]["error_type"], "TimeoutError")
            for filename in ("traj.json", "results.json", "code.py", "results_eval.json"):
                self.assertTrue((result.run_dir / filename).exists())
            self.assertEqual(
                json.loads((result.run_dir / "results.json").read_text(encoding="utf-8")),
                [],
            )

    def test_silent_container_allocation_failure_stops_before_model_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_root = self._write_dataset(root)
            environment = FailedAllocationEnvironment(max_turns=4)
            backend = FakeBackend(["<answer>must not run</answer>"])
            adapter = LongDSAdapter(
                dsgym_root=root / "unused",
                dataset_root=task_root,
                output_root=root / "results",
                model="fake/model",
                max_steps_per_turn=4,
                backend_factory=lambda: backend,
                environment_factory=lambda: environment,
                clean_output=clean_output,
                system_prompt_template="OFFICIAL DATA={PATH}",
            )

            result = adapter.run_task(adapter.load_tasks()[0], manager=None)

            self.assertEqual(result.turn_results, ())
            self.assertEqual(backend.snapshots, [])
            self.assertEqual(result.trajectory["metadata"]["error_type"], "RuntimeError")
            self.assertIn("was not allocated", result.trajectory["error"])

    def test_runner_continues_after_one_task_raises(self):
        calls = []

        class FakeAdapter:
            def __init__(self, **kwargs):
                del kwargs

            def _ensure_runtime_components(self):
                return None

            def load_tasks(self, **kwargs):
                del kwargs
                return (SimpleNamespace(task_key="task1"), SimpleNamespace(task_key="task2"))

            def run_task(self, task, *, manager=None):
                del manager
                calls.append(task.task_key)
                if task.task_key == "task1":
                    raise TimeoutError("simulated task failure")
                return SimpleNamespace(
                    run_dir=Path("/tmp/task2"), trajectory={"error": None}
                )

        args = SimpleNamespace(
            dsgym_root=Path("/tmp/dsgym"),
            dataset_path=Path("/tmp/dataset"),
            output_dir=Path("/tmp/results"),
            model="fake/model",
            backend="litellm",
            manager_url="http://fake",
            max_steps=4,
            temperature=0.0,
            api_key=None,
            base_url=None,
            max_model_len=1024,
            reset_env_times=0,
            task_concurrency=2,
            start_index=0,
            task_limit=None,
            turn_limit=None,
            manager_model=None,
            manager_api_base=None,
            manager_api_key=None,
            manager_file_dir=None,
            manager_file_timeout=7200.0,
            judge=False,
            judge_model="fake-judge",
            judge_api_key=None,
            judge_base_url=None,
        )
        parser = SimpleNamespace(parse_args=lambda: args)
        output = io.StringIO()
        with (
            mock.patch.object(longds_runner, "build_parser", return_value=parser),
            mock.patch.object(longds_runner, "LongDSAdapter", FakeAdapter),
            contextlib.redirect_stdout(output),
        ):
            exit_code = longds_runner.main()

        self.assertEqual(exit_code, 0)
        self.assertCountEqual(calls, ["task1", "task2"])
        self.assertIn("ERROR >>> TASK task1 failed", output.getvalue())
        self.assertIn("RESULT >>> /tmp/task2", output.getvalue())

    def test_hint_is_official_user_information_and_does_not_reset_budget(self):
        environment = FakeEnvironment(max_turns=3)
        workspace = LongDSWorkspace(Path("/tmp/data"))
        backend = FakeBackend(["<answer>wrong</answer>", "<answer>fixed</answer>"])
        worker = LongDSWorkerAgent(
            backend=backend,
            environment=environment,
            workspace=workspace,
            clean_output=clean_output,
            max_steps_per_turn=3,
        )
        worker.start(
            [
                {"role": "system", "content": "official"},
                {"role": "user", "content": "question"},
            ]
        )
        worker.step()
        self.assertTrue(worker.done)
        worker.inject_observation(
            "<error_hint>recheck x</error_hint>",
            metadata={"stateguard": "light_repair"},
        )
        self.assertFalse(worker.done)
        self.assertEqual(worker.turn_step_count, 1)
        self.assertEqual(
            worker.conversation[-1]["content"],
            "<information>\n<error_hint>recheck x</error_hint>\n</information>",
        )
        fixed = worker.step()
        self.assertEqual(fixed.action.answer, "fixed")
        self.assertEqual(worker.turn_step_count, 2)

    def test_last_budget_python_action_is_not_executed_like_official_env(self):
        environment = FakeEnvironment(max_turns=1)
        workspace = LongDSWorkspace(Path("/tmp/data"))
        worker = LongDSWorkerAgent(
            backend=FakeBackend(["<python>x = 1</python>"]),
            environment=environment,
            workspace=workspace,
            clean_output=clean_output,
            max_steps_per_turn=1,
        )
        worker.start(
            [
                {"role": "system", "content": "official"},
                {"role": "user", "content": "question"},
            ]
        )
        step = worker.step()
        self.assertTrue(step.done)
        self.assertEqual(step.metadata["completion_reason"], "budget_exhausted")
        self.assertIsNone(step.observation)
        self.assertEqual(workspace.executions, [])

    def test_workspace_checkpoint_restarts_and_replays_exact_code_prefix(self):
        environment = FakeEnvironment(max_turns=10)
        workspace = LongDSWorkspace(Path("/tmp/data"))
        workspace.bind_environment(environment, clean_output)
        environment.init([])
        environment.tool_group.execute_code("x = 1")
        checkpoint = workspace.snapshot()
        workspace.restore(checkpoint)
        self.assertEqual(environment.tool_group.delegate.client.restarts, 0)
        environment.tool_group.execute_code("x = 2")
        workspace.restore(checkpoint)
        self.assertEqual(environment.tool_group.delegate.client.restarts, 1)
        observed = environment.tool_group.execute_hidden("print(x)")
        self.assertEqual(observed.cleaned_output.strip(), "1")
        environment.tool_group.execute_code("y = 3")
        self.assertEqual(len(workspace.executions), 2)
        workspace.remove_variables(("x",))
        absent = environment.tool_group.execute_hidden("print('x' in globals())")
        self.assertEqual(absent.cleaned_output.strip(), "False")

    def test_manager_probe_uses_fresh_task_data_scratch_and_closes_immediately(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            nested = data_root / "nested"
            nested.mkdir()
            data_path = nested / "input.txt"
            data_path.write_text("checked-data", encoding="utf-8")
            worker_environment = FakeEnvironment(max_turns=10)
            workspace = LongDSWorkspace(data_root)
            workspace.bind_environment(worker_environment, clean_output)
            worker_environment.init([])
            worker_environment.tool_group.execute_code("x = 7")
            probe_environments = []

            def probe_factory():
                environment = FakeEnvironment(max_turns=10)
                probe_environments.append(environment)
                return environment

            probe = LongDSProbeExecutor(workspace, probe_factory, clean_output)
            first = probe.execute(
                "from pathlib import Path\n"
                "assert 'x' not in globals()\n"
                f"assert Path(DATA_ROOT) == Path({str(data_root)!r})\n"
                "scratch_only = 1\n"
                "print(Path(data_files['nested/input.txt']).read_text())"
            )
            self.assertTrue(first.ok)
            self.assertEqual(first.stdout.strip(), "checked-data")
            self.assertTrue(probe_environments[0].closed)
            self.assertIsNone(probe_environments[0].tool_group.allocated_container)
            self.assertEqual(
                worker_environment.tool_group.delegate.client.namespace.get("x"), 7
            )

            (data_root / "created_after_first_probe.txt").write_text(
                "not a benchmark input", encoding="utf-8"
            )
            second = probe.execute(
                "assert 'scratch_only' not in globals()\n"
                "assert 'created_after_first_probe.txt' not in data_files\n"
                "print('fresh')"
            )
            self.assertTrue(second.ok)
            self.assertEqual(second.stdout.strip(), "fresh")
            self.assertTrue(probe_environments[1].closed)

            failed = probe.execute("raise RuntimeError('boom')")
            self.assertFalse(failed.ok)
            self.assertIn("boom", failed.error)
            self.assertTrue(probe_environments[2].closed)
            self.assertIsNone(probe_environments[2].tool_group.allocated_container)
            self.assertEqual(len(probe_environments), 3)
            probe.close()


if __name__ == "__main__":
    unittest.main()
