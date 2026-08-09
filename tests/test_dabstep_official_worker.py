from __future__ import annotations

import unittest
import copy
from dataclasses import dataclass
from pathlib import Path

from stateguard.adapters.dabstep.dataset import DABstepTask
from stateguard.adapters.dabstep.worker import DABstepWorkerAgent


try:
    from smolagents import CodeAgent
    from smolagents.prompts import CODE_SYSTEM_PROMPT
except ImportError:  # Core tests do not require benchmark extras.
    CodeAgent = None
    CODE_SYSTEM_PROMPT = ""


@dataclass
class FakeModelMessage:
    content: str


class ScriptedModel:
    model_id = "fake"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append(copy.deepcopy((messages, kwargs)))
        return FakeModelMessage(self.outputs.pop(0))


@unittest.skipIf(CodeAgent is None, "DABstep optional dependencies are not installed")
class DABstepOfficialWorkerTests(unittest.TestCase):
    def test_native_codeagent_pauses_accepts_feedback_and_keeps_python_state(self):
        model = ScriptedModel(
            [
                "Thought: compute.\nCode:\n```py\nx = 41\nprint(x)\n```<end_code>",
                "Thought: conclude.\nCode:\n```py\nfinal_answer(x + 1)\n```<end_code>",
            ]
        )
        native = CodeAgent(
            tools=[],
            model=model,
            system_prompt=CODE_SYSTEM_PROMPT,
            additional_authorized_imports=[],
            max_steps=2,
            verbosity_level=0,
        )
        task = DABstepTask(
            "test",
            "compute",
            "number only",
            "easy",
            "default",
            Path("/fs/fast/u2024201619/DABstep/data/context"),
        )
        worker = DABstepWorkerAgent(
            task=task,
            native_agent=native,
            model_id="fake",
            max_steps=2,
            prompt_builder=lambda bound_task, agent: "compute",
        )
        worker.start(task.task_spec())
        first = worker.step()
        self.assertFalse(first.done)
        self.assertTrue(first.observation.ok)
        self.assertEqual(native.python_executor.state["x"], 41)
        worker.inject_observation("Re-check the final arithmetic.")
        second = worker.step()
        self.assertTrue(second.done)
        self.assertEqual(worker.final_answer, "42")
        memory = model.calls[1][0]
        self.assertTrue(
            any("<manager_feedback>" in str(message["content"]) for message in memory)
        )


    def test_manager_none_facade_matches_direct_official_run(self):
        outputs = [
            "Thought: compute.\nCode:\n```py\nx = 41\nprint(x)\n```<end_code>",
            "Thought: conclude.\nCode:\n```py\nfinal_answer(x + 1)\n```<end_code>",
        ]
        direct_model = ScriptedModel(outputs)
        direct = CodeAgent(
            tools=[],
            model=direct_model,
            system_prompt=CODE_SYSTEM_PROMPT,
            additional_authorized_imports=[],
            max_steps=2,
            verbosity_level=0,
        )
        expected = direct.run("compute")

        facade_model = ScriptedModel(outputs)
        native = CodeAgent(
            tools=[],
            model=facade_model,
            system_prompt=CODE_SYSTEM_PROMPT,
            additional_authorized_imports=[],
            max_steps=2,
            verbosity_level=0,
        )
        task = DABstepTask(
            "test",
            "compute",
            "number only",
            "easy",
            "default",
            Path("/fs/fast/u2024201619/DABstep/data/context"),
        )
        worker = DABstepWorkerAgent(
            task=task,
            native_agent=native,
            model_id="fake",
            max_steps=2,
            prompt_builder=lambda bound_task, agent: "compute",
        )
        worker.start(task.task_spec())
        while not worker.done:
            worker.step()

        self.assertEqual(worker.final_answer, str(expected))
        self.assertEqual(native.python_executor.state["x"], direct.python_executor.state["x"])
        self.assertEqual(facade_model.calls, direct_model.calls)


if __name__ == "__main__":
    unittest.main()
