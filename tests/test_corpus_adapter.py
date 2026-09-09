from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stateguard.adapters.corpus import runner as corpus_runner
from stateguard.adapters.corpus.adapter import CorpusAdapter
from stateguard.adapters.corpus.dataset import IDABenchV2Loader
from stateguard.agents.manager import StateManagerAgent
from stateguard.providers.base import ModelResponse


class FakeResponse:
    status_code = 200
    is_success = True

    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        return None

    def json(self):
        return self.value


class FakeClient:
    def __init__(self):
        self.namespace = {}

    def post(self, url, json=None):
        if url.endswith("/restart"):
            self.namespace = {}
            return FakeResponse({"status": "ok"})
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile((json or {})["code"], "<corpus-test>", "exec"), self.namespace)
            outputs = ([{"type": "stream", "text": stdout.getvalue()}]
                       if stdout.getvalue() else [])
        except Exception as exc:
            outputs = [{"type": "error", "name": type(exc).__name__, "value": str(exc)}]
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

    def deallocate_container(self):
        self.allocated_container = None

    def get_tool_names(self):
        return ["python"]


class FakeEnvironment:
    def __init__(self):
        self.tool_group = FakeToolGroup()
        self.chat_history = []
        self.turns = 0
        self.init_calls = 0
        self.reset_calls = 0

    def init(self, messages, **kwargs):
        del kwargs
        self.init_calls += 1
        self.turns = 0
        self.chat_history = [dict(message) for message in messages]
        self.tool_group.allocate_container()
        return [dict(message) for message in messages], {}

    def reset_turns(self):
        self.reset_calls += 1
        self.turns = 0

    def step(self, raw):
        self.turns += 1
        answer = raw.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        return {
            "postprocessed_action": raw,
            "observations": [],
            "done": True,
            "metadata": {"final_answer": answer},
        }

    def close(self):
        self.tool_group.deallocate_container()


class FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, messages):
        del messages
        return self.responses.pop(0)


class QueueModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def complete(self, messages, tools):
        del messages, tools
        decision = self.decisions.pop(0)
        content = json.dumps(
            {
                "type": "control",
                "reasoning": "follow the lifecycle",
                "answer": json.dumps(decision),
            }
        )
        return ModelResponse(content, "follow the lifecycle")


def clean_output(outputs):
    return "".join(str(row.get("text", "")) for row in outputs)


class CorpusAdapterTests(unittest.TestCase):
    def test_ida_loader_keeps_gold_outside_public_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "case"
            (task / "data").mkdir(parents=True)
            (task / "data" / "train.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            (task / "task.json").write_text(
                json.dumps([
                    {"turn_id": 1, "relation": "init", "upstream": [],
                     "source_shards": [1], "question": "Compute y."}
                ]), encoding="utf-8"
            )
            (task / "answers.json").write_text(
                json.dumps({"turns": [{"turn_id": 1, "answer": {"y": 2}}]}),
                encoding="utf-8",
            )
            loaded = IDABenchV2Loader(root).load()[0].units[0]
            public = loaded.public.task_spec()
            self.assertEqual(public.data_files, (str((task / "data" / "train.csv").resolve()),))
            self.assertNotIn("answer", json.dumps(public.metadata))
            self.assertNotIn("relation", json.dumps(public.metadata))
            self.assertEqual(loaded.private.reference_answer, {"y": 2})
            self.assertEqual(loaded.private.expected_relation, "init")

    def test_multi_turn_reuses_one_worker_conversation_and_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = _dsbench_fixture(root, multi=True)
            environment = FakeEnvironment()
            backend = FakeBackend([
                "<reasoning>done one</reasoning><answer>one</answer>",
                "<reasoning>done two</reasoning><answer>two</answer>",
            ])
            adapter = _adapter(root, corpus, environment, backend, mode="multi_turn")
            task = adapter.load_tasks()[0]
            result = adapter.run_task(task, run_dir=root / "run-multi")
            self.assertIsNone(result.error)
            self.assertEqual([row["solution"] for row in result.unit_results], ["one", "two"])
            self.assertEqual(environment.init_calls, 1)
            self.assertEqual(environment.reset_calls, 1)
            self.assertEqual(result.trajectory["total_worker_steps"], 2)

    def test_empty_solution_under_budget_remains_sft_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = _dsbench_fixture(root, multi=True)
            adapter = _adapter(
                root,
                corpus,
                FakeEnvironment(),
                FakeBackend([
                    "<reasoning>submitted no answer</reasoning><answer></answer>",
                    "<reasoning>done two</reasoning><answer>two</answer>",
                ]),
                mode="multi_turn",
            )
            result = adapter.run_task(adapter.load_tasks()[0], run_dir=root / "run")
            first = result.unit_results[0]
            self.assertEqual(first["solution"], "")
            self.assertFalse(first["worker_budget_exhausted"])
            self.assertTrue(first["sft_eligible"])

    def test_single_query_exports_valid_manager_activations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = _dsbench_fixture(root, multi=False)
            environment = FakeEnvironment()
            backend = FakeBackend([
                "<reasoning>computed</reasoning><answer>42</answer>"
            ])
            adapter = _adapter(root, corpus, environment, backend, mode="single_query")
            manager = StateManagerAgent(QueueModel([
                {"action": "OPEN_STATE", "state_header": {
                    "constraints": [{"text": "Answer the query."}]
                }},
                {"action": "UPDATE_STATE", "state_update": {
                    "issue": "Compute the requested result.",
                    "used_variables": [{"name": "answer", "value": 42}],
                    "conclusions": ["The Worker returned the requested result."],
                    "source_interval": {"start": 1, "end": 1}
                }},
                {"action": "FINALIZE_RELATIONS", "relation_finalization": {
                    "mode": "select", "relations": [{"type": "init"}]
                }},
                {"action": "COMMIT_STATE"},
            ]))
            result = adapter.run_task(
                adapter.load_tasks()[0], manager=manager, run_dir=root / "run-single"
            )
            self.assertIsNone(result.error)
            records = json.loads(
                (result.run_dir / "sft_data_activations.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(records), 4)
            self.assertEqual(
                result.stateguard_results[0].committed_states[0]
                .used_variables[0].version,
                "S1",
            )
            self.assertTrue(all(row["messages"][0]["role"] == "system" for row in records))
            manager_text = json.dumps(records)
            self.assertNotIn("reference_answer", manager_text)
            self.assertNotIn('"answer": 42', manager_text)


    def test_ida_runtime_stages_only_declared_data_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "IDA-Bench-v2"
            task_dir = corpus / "case"
            nested = task_dir / "data" / "nested"
            nested.mkdir(parents=True)
            source = nested / "train.csv"
            source.write_text("x,y\n1,2\n", encoding="utf-8")
            (task_dir / "task.json").write_text(
                json.dumps([{"turn_id": 1, "question": "Compute y."}]),
                encoding="utf-8",
            )
            (task_dir / "answers.json").write_text(
                json.dumps({"turns": [{"turn_id": 1, "answer": {"y": 2}}]}),
                encoding="utf-8",
            )
            (task_dir / "ground_truth").mkdir()
            (task_dir / "ground_truth" / "answer.csv").write_text(
                "y\n2\n", encoding="utf-8"
            )
            dsgym = root / "dsgym"
            dsgym.mkdir()
            adapter = CorpusAdapter(
                source="idabench_v2",
                corpus_root=corpus,
                dsgym_root=dsgym,
                output_root=root / "outputs",
                model="fake",
                mode="multi_turn",
                system_prompt_template="Data are in {PATH}.",
                backend_factory=lambda: FakeBackend([]),
                environment_factory=FakeEnvironment,
                clean_output=clean_output,
            )
            task = adapter.load_tasks()[0]
            run_dir = root / "run"
            staged_once = adapter._runtime_task(task, run_dir)
            staged_twice = adapter._runtime_task(task, run_dir)

            staged_file = run_dir / "worker_data" / "nested" / "train.csv"
            self.assertTrue(staged_file.is_file())
            self.assertEqual(staged_once.data_root, run_dir / "worker_data")
            self.assertEqual(staged_twice.units[0].public.data_files, (staged_file.resolve(),))
            self.assertFalse((run_dir / "answers.json").exists())
            self.assertFalse((run_dir / "ground_truth").exists())
            public_text = json.dumps(staged_once.units[0].public.task_spec().metadata)
            self.assertNotIn(str(task_dir), public_text)

    def test_staging_rejects_a_changed_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = _dsbench_fixture(root, multi=False)
            adapter = _adapter(
                root,
                corpus,
                FakeEnvironment(),
                FakeBackend(["<answer>42</answer>"]),
                mode="single_query",
            )
            task = adapter.load_tasks()[0]
            run_dir = root / "run"
            adapter._runtime_task(task, run_dir)
            destination = run_dir / "worker_data" / "data.csv"
            destination.unlink()
            destination.write_text("different\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                adapter._runtime_task(task, run_dir)

    def test_empty_manager_session_is_reported_without_fake_sft_file(self):
        class EmptyManager:
            def export_session(self):
                return {"messages": []}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = _dsbench_fixture(root, multi=False)
            adapter = _adapter(
                root,
                corpus,
                FakeEnvironment(),
                FakeBackend([]),
                mode="single_query",
            )
            run_dir = root / "run"
            adapter._write_manager_and_sft(
                run_dir=run_dir,
                manager=EmptyManager(),
                manager_failures=[],
            )
            report = json.loads(
                (run_dir / "sft_export_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "skipped_no_activations")
            self.assertFalse((run_dir / "sft_data_activations.json").exists())

    def test_preflight_checks_execution_and_always_deallocates(self):
        responses = iter(
            [
                {"status": "ok"},
                {"available_containers": 4},
                {"container_id": 2},
                {"ready": True},
                {"outputs": [{"text": "stateguard_preflight_ok"}]},
                {"status": "deallocated"},
            ]
        )
        calls = []

        def fake_request(url, **kwargs):
            calls.append((url, kwargs))
            return next(responses)

        with mock.patch.object(corpus_runner, "_request_json", side_effect=fake_request):
            corpus_runner.preflight_dsgym("http://localhost:5000", required_slots=4)
        self.assertTrue(calls[-1][0].endswith("/deallocate/2"))

    def test_runner_returns_failure_and_excludes_failed_task_from_sft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = SimpleNamespace(task_key="failed-task")
            result = SimpleNamespace(
                task=task,
                run_dir=root / "task-run",
                trajectory={"success": False},
                error="allocation failed",
            )

            class FakeAdapter:
                def __init__(self, **kwargs):
                    del kwargs
                    self.run_root = root / "aggregate"

                def load_tasks(self, **kwargs):
                    del kwargs
                    return (task,)

                def run_task(self, selected, *, manager=None):
                    del selected, manager
                    return result

            args = SimpleNamespace(
                source="dsbench_v1",
                corpus_root=root,
                dsgym_root=root,
                output_dir=root,
                experiment="test",
                mode="multi-turn",
                task_id=None,
                start_index=0,
                task_limit=None,
                unit_limit=None,
                task_concurrency=1,
                worker_model="fake",
                worker_backend="litellm",
                worker_api_base=None,
                worker_api_key=None,
                worker_max_steps=4,
                worker_max_tokens=None,
                worker_timeout=None,
                worker_max_retries=None,
                temperature=0.0,
                max_model_len=1024,
                manager_url="http://fake",
                review_cadence=3,
                manager_model=None,
                manager_api_base=None,
                manager_api_key=None,
                manager_file_dir=None,
                manager_file_timeout=1.0,
                allow_partial=False,
            )
            parser = SimpleNamespace(parse_args=lambda: args)
            with (
                mock.patch.object(corpus_runner, "build_parser", return_value=parser),
                mock.patch.object(corpus_runner, "CorpusAdapter", FakeAdapter),
                mock.patch.object(corpus_runner, "preflight_dsgym"),
            ):
                exit_code = corpus_runner.main()
            self.assertEqual(exit_code, 1)
            summaries = list((root / "aggregate" / "exports").glob("*/run_summary.json"))
            self.assertEqual(len(summaries), 1)
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertEqual(summary["failed_tasks"], 1)
            self.assertEqual(summary["sft_records"], 0)
            self.assertEqual(summary["tasks"][0]["sft_status"], "skipped_failed_task")

            # A completed Worker run is still a failed corpus/SFT task when an
            # enabled Manager produced no activation artifact. Raw artifacts stay
            # available, but the aggregate must not silently accept the task.
            result.trajectory = {"success": True}
            result.error = None
            args.manager_file_dir = root / "manager-handshake"
            with (
                mock.patch.object(corpus_runner, "build_parser", return_value=parser),
                mock.patch.object(corpus_runner, "CorpusAdapter", FakeAdapter),
                mock.patch.object(corpus_runner, "preflight_dsgym"),
                mock.patch.object(corpus_runner, "_create_manager", return_value=object()),
            ):
                exit_code = corpus_runner.main()
            self.assertEqual(exit_code, 1)
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertEqual(summary["failed_tasks"], 1)
            self.assertEqual(summary["sft_records"], 0)
            self.assertEqual(summary["tasks"][0]["sft_status"], "skipped_no_activations")
            self.assertIn("no activation", summary["tasks"][0]["artifact_error"])


def _dsbench_fixture(root: Path, *, multi: bool) -> Path:
    corpus = root / "DSBench-v1"
    mode_root = corpus / ("multi-turn" if multi else "single-turn")
    task = mode_root / "task1"
    task.mkdir(parents=True)
    (task / "introduction.txt").write_text("Use the supplied data.", encoding="utf-8")
    (task / "data.csv").write_text("x\n1\n", encoding="utf-8")
    if multi:
        (task / "question2.txt").write_text("First question?", encoding="utf-8")
        (task / "question10.txt").write_text("Second question?", encoding="utf-8")
        answers = {"task1": {"questions": ["question2", "question10"],
                             "answers": ["one", "two"]}}
    else:
        (task / "question.txt").write_text("Return the result.", encoding="utf-8")
        answers = {"task1": {"format": "plain", "answer": "42"}}
    (mode_root / "answers.json").write_text(json.dumps(answers), encoding="utf-8")
    return corpus


def _adapter(root, corpus, environment, backend, *, mode):
    dsgym = root / "dsgym"
    dsgym.mkdir()
    return CorpusAdapter(
        source="dsbench_v1",
        corpus_root=corpus,
        dsgym_root=dsgym,
        output_root=root / "outputs",
        model="fake",
        mode=mode,
        max_worker_steps=5,
        system_prompt_template="Data are in {PATH}.",
        backend_factory=lambda: backend,
        environment_factory=lambda: environment,
        clean_output=clean_output,
    )


if __name__ == "__main__":
    unittest.main()
