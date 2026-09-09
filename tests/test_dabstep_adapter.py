from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from stateguard.adapters.dabstep.adapter import DABstepAdapter
from stateguard.adapters.dabstep.dataset import CONTEXT_FILENAMES, DABstepDataset
from stateguard.adapters.dabstep.workflow import DABstepWorkflow
from stateguard.core.events import ReActStep
from stateguard.core.models import AgentAction, Message, ToolResult
from stateguard.state.draft import (
    RelationFinalization,
    RelationFinalizationMode,
    SourceInterval,
    StateDraft,
    StateHeader,
)
from stateguard.state.models import Constraint, StateRelation, StateRelationType

try:  # Only the official-runtime test needs the DABstep extras.
    import smolagents
except ImportError:
    smolagents = None


class FakeWorker:
    def __init__(self):
        self._messages = []
        self._done = False
        self._answer = None
        self.accepted_steps = 0

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
        self._messages.append(Message("user", prompt.query))

    def continue_turn(self, prompt):
        raise AssertionError(prompt)

    def step(self):
        self.accepted_steps += 1
        self._done = True
        self._answer = "NL"
        return ReActStep(
            self.accepted_steps,
            AgentAction(kind="final", answer="NL"),
            None,
            True,
            raw_model_output="final_answer('NL')",
        )

    def inject_observation(self, content, metadata=None):
        self._messages.append(Message("user", str(content), metadata=metadata or {}))
        self._done = False

    def snapshot(self):
        return copy.deepcopy(
            (self._messages, self._done, self._answer, self.accepted_steps)
        )

    def restore(self, snapshot):
        self._messages, self._done, self._answer, self.accepted_steps = copy.deepcopy(
            snapshot
        )

    def trajectory(self):
        return [{"step": 0, "action_output": "NL"}]

    def close(self):
        return None


class DABstepAdapterTests(unittest.TestCase):
    def test_dataset_counts_and_gold_firewall_on_local_official_data(self):
        root = Path("/fs/fast/u2024201619/DABstep")
        dataset = DABstepDataset(root)
        self.assertEqual(len(dataset.load("default")), 450)
        self.assertEqual(len(dataset.load("dev")), 10)
        task = dataset.load("dev", task_ids=[5])[0]
        self.assertEqual(task.reference_answer, "NL")
        rendered = json.dumps(task.task_spec().metadata).lower()
        self.assertNotIn("answer", rendered)
        self.assertNotIn("gold", rendered)
        self.assertNotIn("reference", rendered)

    @unittest.skipIf(smolagents is None, "DABstep optional dependencies are not installed")
    def test_default_runtime_profile_is_compat_and_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = DABstepAdapter(
                dabstep_root=Path("/fs/fast/u2024201619/DABstep"),
                output_root=Path(tmp),
                model_id="fake/model",
                split="dev",
            )
            self.assertEqual(adapter.runtime_profile, "compat-v1")
            self.assertEqual(adapter.runtime_report["runtime_profile"], "compat-v1")
            self.assertEqual(adapter.runtime_report["smolagents_version"], "1.3.0")
            self.assertTrue(adapter.runtime_report["fixes"])
            config = (adapter.run_root / "config.yaml").read_text(encoding="utf-8")
            self.assertIn("compat-v1", config)
            self.assertIn("dabstep-smolagents-1.3-compat-v1", config)

    def test_workflow_uses_three_step_pause_no_hint_and_posthoc_relations(self):
        workflow = DABstepWorkflow()
        lifecycle = workflow.lifecycle_prompt()
        self.assertIn("every 3 accepted official Worker action steps", lifecycle)
        self.assertIn("not a state boundary", lifecycle)
        self.assertIn("relations posthoc", lifecycle)
        step = ReActStep(
            3,
            AgentAction(kind="tool", tool_name="python_interpreter"),
            ToolResult("python_interpreter", True, "ok"),
            False,
        )
        self.assertFalse(workflow.should_review(step, 2))
        self.assertTrue(workflow.should_review(step, 3))
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
        draft = StateDraft(StateHeader("S2", (Constraint("query"),), ()))
        draft.issue = "computed total"
        draft.source_interval = SourceInterval(1, 2)
        workflow.validate_relation_finalization(
            draft=draft,
            finalization=RelationFinalization(
                RelationFinalizationMode.SELECT,
                (StateRelation(StateRelationType.INIT),),
                "no prior state",
            ),
            untraced_steps=(step,),
        )

    def test_manager_none_writes_official_answer_format_without_manager_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = _fake_dataset(base / "DABstep")
            official = root / "dabstep_official_runner"
            official.mkdir()
            adapter = DABstepAdapter(
                dabstep_root=root,
                official_runner_root=official,
                output_root=base / "runs",
                model_id="fake/model",
                split="default",
                worker_factory=lambda task, workspace: FakeWorker(),
            )
            result = adapter.run_task(adapter.load_tasks()[0], manager=None)
            self.assertIsNone(result.error)
            self.assertEqual(result.stateguard_result.manager_actions, 0)
            self.assertEqual(result.benchmark_result["agent_answer"], "NL")
            rows = [
                json.loads(line)
                for line in (adapter.run_root / "answers.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(rows, [{"task_id": "5", "agent_answer": "NL"}])
            self.assertTrue((result.run_dir / "trajectory.json").is_file())


def _fake_dataset(root: Path) -> Path:
    context = root / "data" / "context"
    tasks = root / "data" / "tasks"
    context.mkdir(parents=True)
    tasks.mkdir(parents=True)
    for name in CONTEXT_FILENAMES:
        (context / name).write_text("x", encoding="utf-8")
    row = {
        "task_id": "5",
        "question": "Which country?",
        "answer": "",
        "guidelines": "country code only",
        "level": "easy",
    }
    (tasks / "all.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    dev = {**row, "answer": "NL"}
    (tasks / "dev.jsonl").write_text(json.dumps(dev) + "\n", encoding="utf-8")
    return root


if __name__ == "__main__":
    unittest.main()
