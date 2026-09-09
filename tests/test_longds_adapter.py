import contextlib
import time
from concurrent.futures import ThreadPoolExecutor
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
        self.observations = []

    def start_task(self, task):
        self.tasks.append(task)

    def act(self, observation):
        self.observations.append(observation)
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
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    conclusions=(Conclusion("Q1 answer is 41."),),
                ),
            )
        if index == 3:
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    RelationFinalizationMode.CONFIRM,
                    (StateRelation(StateRelationType.INIT),),
                    "First state remains initialization.",
                ),
            )
        if index == 4:
            return ManagerDecision(action=ManagerAction.COMMIT_STATE)
        if index == 5:
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    id="S2",
                    issue="Q2",
                    constraints=(Constraint("Answer Q2 using relevant checked work."),),
                    relations=(StateRelation(StateRelationType.PROGRESS, "S1"),),
                ),
            )
        if index == 6:
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    conclusions=(Conclusion("Q2 answer is 42."),),
                ),
            )
        if index == 7:
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    RelationFinalizationMode.CONFIRM,
                    (StateRelation(StateRelationType.PROGRESS, "S1"),),
                    "The completed turn directly progresses from S1.",
                ),
            )
        return ManagerDecision(action=ManagerAction.COMMIT_STATE)


class FailFirstPreopenManager:
    """Fail the first mandatory OPEN, then manage the second turn normally."""

    def __init__(self):
        self.current_task_id = ""
        self.second_turn_actions = 0
        self.observations = []

    def start_task(self, task):
        self.current_task_id = task.id

    def act(self, observation):
        self.observations.append(observation)
        if self.current_task_id.endswith("turn_1"):
            return ManagerDecision(action=ManagerAction.RESUME_WORKER)
        self.second_turn_actions += 1
        if self.second_turn_actions == 1:
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    id="S2",
                    issue="Q2",
                    constraints=(Constraint("Answer Q2."),),
                    relations=(StateRelation(StateRelationType.INIT),),
                ),
            )
        if self.second_turn_actions == 2:
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    conclusions=(Conclusion("Q2 answer is 42."),),
                ),
            )
        if self.second_turn_actions == 3:
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    RelationFinalizationMode.CONFIRM,
                    (StateRelation(StateRelationType.INIT),),
                    "The skipped predecessor left no committed dependency.",
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
        self.assertIn("binds the complete turn automatically", lifecycle)
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
            self.assertEqual([item.manager_actions for item in result.stateguard_results], [4, 4])
            self.assertEqual(manager.observations[0].committed_state_index, ())
            self.assertIsNone(manager.observations[1].committed_state_index)
            self.assertEqual(manager.observations[2].committed_state_index, ())
            self.assertEqual(
                [item["id"] for item in manager.observations[4].committed_state_index],
                ["S1"],
            )
            self.assertEqual(
                [item["id"] for item in manager.observations[6].committed_state_index],
                ["S1"],
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

    def test_failed_mandatory_open_skips_only_that_turn_and_consumes_state_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_root = self._write_dataset(root)
            adapter = LongDSAdapter(
                dsgym_root=root / "unused",
                dataset_root=task_root,
                output_root=root / "results",
                model="fake/model",
                max_steps_per_turn=4,
                backend_factory=lambda: FakeBackend(
                    ["<answer>41</answer>", "<answer>42</answer>"]
                ),
                environment_factory=lambda: FakeEnvironment(max_turns=4),
                clean_output=clean_output,
                system_prompt_template="OFFICIAL DATA={PATH}",
            )
            manager = FailFirstPreopenManager()

            result = adapter.run_task(adapter.load_tasks()[0], manager=manager)

            first, second = result.stateguard_results
            self.assertEqual(first.final_answer, "41")
            self.assertTrue(first.degraded)
            self.assertEqual(first.manager_actions, 1)
            self.assertEqual(first.committed_states, ())
            self.assertEqual(second.final_answer, "42")
            self.assertFalse(second.degraded)
            self.assertEqual(second.manager_actions, 4)
            self.assertEqual([state.id for state in second.committed_states], ["S2"])
            self.assertEqual(len(manager.observations), 5)
            self.assertEqual(manager.observations[1].available_state_id, "S2")
            state_events = [
                json.loads(line)
                for line in (result.run_dir / "stateguard" / "state.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(
                any(
                    row.get("event") == "mandatory_preopen_failed"
                    and row.get("state_id") == "S1"
                    for row in state_events
                )
            )

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

    def test_runtime_components_initialize_once_under_concurrency(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_root = self._write_dataset(root)
            adapter = LongDSAdapter(
                dsgym_root=root / "unused",
                dataset_root=task_root,
                output_root=root / "results",
                model="fake/model",
                system_prompt_template="OFFICIAL DATA={PATH}",
            )
            calls = 0

            def official_components():
                nonlocal calls
                calls += 1
                time.sleep(0.02)
                return (lambda: object(), lambda: object(), lambda outputs: "")

            with mock.patch.object(
                adapter, "_official_components", side_effect=official_components
            ):
                with ThreadPoolExecutor(max_workers=8) as executor:
                    list(executor.map(lambda _: adapter._ensure_runtime_components(), range(16)))

            self.assertEqual(calls, 1)
            self.assertTrue(adapter._runtime_components_ready())

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
            worker_timeout=None,
            worker_max_retries=None,
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


class StateGuardV6LoaderTest(unittest.TestCase):
    """The v6 release is single-query only and carries its output contract."""

    ROOT = Path("/fs/fast/u2024201619/StateGuard-SFT-Corpus-v6-20")

    def setUp(self) -> None:
        if not self.ROOT.is_dir():
            self.skipTest("StateGuard SFT corpus v6 is not installed")

    def test_single_query_tasks_expose_question_guidelines_and_staged_files(self):
        from stateguard.adapters.corpus.dataset import create_corpus_loader

        loader = create_corpus_loader("stateguard_v6", self.ROOT)
        tasks = loader.load(mode="single_query")
        self.assertTrue(tasks)
        for task in tasks:
            self.assertEqual(task.mode, "single_query")
            self.assertEqual(len(task.units), 1)
            spec = task.units[0].public.task_spec()
            self.assertEqual(spec.id, f"stateguard_v6/{task.raw_task_id}")
            self.assertTrue(spec.query.strip())
            self.assertTrue(spec.context.strip())
            self.assertTrue(spec.data_files)
            for path in spec.data_files:
                self.assertTrue(Path(path).is_file())
            self.assertIsNotNone(task.units[0].private.reference_answer)

    def test_multi_turn_is_rejected_rather_than_silently_empty(self):
        from stateguard.adapters.corpus.dataset import create_corpus_loader

        loader = create_corpus_loader("stateguard_v6", self.ROOT)
        with self.assertRaises(ValueError):
            loader.load(mode="multi_turn")

    def test_other_corpora_stay_free_of_guidelines(self):
        """Guidelines are additive; the older sources must render as before."""
        from stateguard.adapters.corpus.dataset import create_corpus_loader

        root = Path("/fs/fast/u2024201619/DSBench/DSBench-v1")
        if not root.is_dir():
            self.skipTest("DSBench-v1 is not installed")
        loader = create_corpus_loader("dsbench_v1", root)
        for task in loader.load(mode="single_query"):
            self.assertEqual(task.units[0].public.task_spec().guidelines, ())


class StateGuardMediumSingleLoaderTest(unittest.TestCase):
    """The medium single-query release renders its answer contract for the model."""

    ROOT = Path("/fs/fast/u2024201619/StateGuard-SFT-SingleQuery-Medium-v2")

    def setUp(self) -> None:
        if not self.ROOT.is_dir():
            self.skipTest("StateGuard medium single-query corpus is not installed")

    def test_tasks_load_with_rendered_contract_and_staged_files(self):
        from stateguard.adapters.corpus.dataset import create_corpus_loader

        loader = create_corpus_loader("stateguard_medium_single", self.ROOT)
        tasks = loader.load(mode="single_query")
        self.assertTrue(tasks)
        for task in tasks:
            self.assertEqual(len(task.units), 1)
            spec = task.units[0].public.task_spec()
            self.assertEqual(spec.id, f"stateguard_medium_single/{task.raw_task_id}")
            self.assertTrue(spec.query.strip())
            self.assertTrue(spec.data_files)
            for path in spec.data_files:
                self.assertTrue(Path(path).is_file())
            # analysis_rules.md carries the metric definitions and must be staged.
            self.assertTrue(
                any(p.endswith("analysis_rules.md") for p in spec.data_files)
            )

    def test_staged_files_are_exactly_the_declared_list(self):
        """SQLite drops -wal/-shm sidecars beside a database it opens.

        A directory walk would stage those as task inputs and would make the
        staged set depend on whether anything had recently read the database, so
        the loader must follow the task's declared file list instead.
        """
        import json as _json

        from stateguard.adapters.corpus.dataset import create_corpus_loader

        loader = create_corpus_loader("stateguard_medium_single", self.ROOT)
        for task in loader.load(mode="single_query"):
            task_dir = self.ROOT / "tasks" / task.raw_task_id
            declared = _json.loads(
                (task_dir / "question.json").read_text(encoding="utf-8")
            )["files"]
            expected = sorted(str((task_dir / item).resolve()) for item in declared)
            staged = sorted(task.units[0].public.task_spec().data_files)
            self.assertEqual(staged, expected)
            for path in staged:
                self.assertFalse(path.endswith(("-wal", "-shm")), path)

    def test_a_file_outside_the_data_root_is_refused(self):
        from stateguard.adapters.corpus.dataset import _declared_data_files

        task_dir = self.ROOT / "tasks" / "medium_sq_014"
        with self.assertRaises(ValueError):
            _declared_data_files(task_dir, task_dir / "files", ["../question.json"])
        with self.assertRaises(FileNotFoundError):
            _declared_data_files(task_dir, task_dir / "files", ["files/absent.csv"])
        with self.assertRaises(ValueError):
            _declared_data_files(
                task_dir,
                task_dir / "files",
                ["files/analysis_rules.md", "files/analysis_rules.md"],
            )
            joined = "\n".join(spec.guidelines)
            self.assertIn("exactly these keys", joined)
            self.assertIsInstance(task.units[0].private.reference_answer, dict)

    def test_multi_turn_is_rejected(self):
        from stateguard.adapters.corpus.dataset import create_corpus_loader

        loader = create_corpus_loader("stateguard_medium_single", self.ROOT)
        with self.assertRaises(ValueError):
            loader.load(mode="multi_turn")

    def test_contract_rendering_handles_a_plain_list(self):
        """A list-shaped guidelines block passes through unchanged."""
        from stateguard.adapters.corpus.dataset import _render_answer_contract

        self.assertEqual(_render_answer_contract(["a", "b"]), ("a", "b"))
        self.assertEqual(_render_answer_contract(None), ())

    def test_answer_contract_reaches_the_worker_context(self):
        """The DSGym turn prompt is context + question, with no guidelines slot.

        This release states its required keys only in `guidelines`, so a
        contract left there alone would reach the Manager and never the Worker.
        """
        from stateguard.adapters.corpus.dataset import create_corpus_loader

        loader = create_corpus_loader("stateguard_medium_single", self.ROOT)
        for task in loader.load(mode="single_query"):
            spec = task.units[0].public.task_spec()
            self.assertIn("Answer contract:", spec.context)
            for line in spec.guidelines:
                self.assertIn(line, spec.context)

    def test_other_corpora_keep_their_context_untouched(self):
        from stateguard.adapters.corpus.dataset import create_corpus_loader

        root = Path("/fs/fast/u2024201619/DSBench/DSBench-v1")
        if not root.is_dir():
            self.skipTest("DSBench-v1 is not installed")
        loader = create_corpus_loader("dsbench_v1", root)
        for task in loader.load(mode="single_query"):
            self.assertNotIn("Answer contract:", task.units[0].public.task_spec().context)


class AssistantActionNormalisationTest(unittest.TestCase):
    """The exporter must accept what the runtime accepted, in the runtime's shape."""

    def test_a_narrated_action_is_recovered_as_the_serialized_object(self):
        from stateguard.sft.activation import _normalize_assistant_action

        narrated = (
            "I've verified the computation. Now I'll form the state.\n\n"
            '{"type":"control","reasoning":"ok","answer":{"action":"RESUME_WORKER"}}'
        )
        recovered = _normalize_assistant_action(narrated)
        self.assertEqual(
            json.loads(recovered),
            {
                "type": "control",
                "reasoning": "ok",
                "answer": {"action": "RESUME_WORKER"},
            },
        )
        self.assertFalse(recovered.lstrip().startswith("I've"))

    def test_a_clean_action_is_returned_untouched(self):
        from stateguard.sft.activation import _normalize_assistant_action

        clean = '{"type": "control", "reasoning": "ok", "answer": {"action": "RESUME_WORKER"}}'
        self.assertEqual(_normalize_assistant_action(clean), clean)

    def test_text_without_any_object_still_raises(self):
        from stateguard.agents.react import ActionParseError
        from stateguard.sft.activation import _normalize_assistant_action

        with self.assertRaises(ActionParseError):
            _normalize_assistant_action("no action at all")
