from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from stateguard.adapters.dacomp.workflow import DACompWorkflow
from stateguard.core.events import ReActStep
from stateguard.core.models import AgentAction, Message, TaskSpec, ToolResult
from stateguard.harness.engine import StateGuardConfig, StateGuardHarness
from stateguard.runtime.bundle import StateGuardRuntime
from stateguard.runtime.workspace import InMemoryWorkspace
from stateguard.state.draft import (
    RelationFinalization,
    RelationFinalizationMode,
    StateHeader,
    StateUpdate,
)
from stateguard.state.models import Conclusion, Constraint, StateRelation, StateRelationType
from stateguard.validation.models import ManagerAction, ManagerDecision


class SixStepWorker:
    def __init__(self):
        self._messages = []
        self.index = 0
        self._done = False
        self._answer = None
        self.injected = []

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
        self._messages.append(Message("user", str(prompt)))

    def continue_turn(self, prompt):
        raise AssertionError(prompt)

    def step(self):
        self.index += 1
        terminal = self.index == 6
        if terminal:
            self._done = True
            self._answer = "answer"
            action = AgentAction(kind="final", answer="answer")
            observation = None
        else:
            action = AgentAction(kind="tool", tool_name="python", arguments={"code": "x=1"})
            observation = ToolResult(
                "python",
                True,
                "ok",
                {"execution_attempted": True, "execution_succeeded": True},
            )
        return ReActStep(self.index, action, observation, terminal)

    def inject_observation(self, content, metadata=None):
        self.injected.append((content, metadata))
        self._messages.append(Message("user", str(content)))
        self._done = False

    def snapshot(self):
        return copy.deepcopy((self._messages, self.index, self._done, self._answer, self.injected))

    def restore(self, snapshot):
        self._messages, self.index, self._done, self._answer, self.injected = copy.deepcopy(snapshot)


class PrefixManager:
    def __init__(self):
        self.calls = 0

    def configure_lifecycle(self, prompt):
        self.lifecycle = prompt

    def start_task(self, task):
        self.task = task

    def act(self, observation):
        self.calls += 1
        if self.calls == 1:
            self.assert_step_ids(observation, (1, 2, 3, 4, 5))
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    id="S1", constraints=(Constraint("answer the query"),), relations=()
                ),
            )
        if self.calls == 2:
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    issue="formed result",
                    conclusions=(Conclusion("steps 1-3 form the result"),),
                    traced_step_ids=(1, 2, 3),
                ),
            )
        if self.calls == 3:
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    RelationFinalizationMode.SELECT,
                    (StateRelation(StateRelationType.INIT),),
                    "no suitable committed predecessor",
                ),
            )
        if self.calls == 4:
            return ManagerDecision(action=ManagerAction.COMMIT_STATE)
        if self.calls == 5:
            return ManagerDecision(action=ManagerAction.RESUME_WORKER)
        self.assert_step_ids(observation, (4, 5, 6))
        return ManagerDecision(action=ManagerAction.ABSTAIN)

    @staticmethod
    def assert_step_ids(observation, expected):
        assert tuple(step.step_id for step in observation.untraced_steps) == expected


class DACompManagedFlowTests(unittest.TestCase):
    def test_five_step_pause_allows_three_step_state_and_injects_no_state_hint(self):
        worker = SixStepWorker()
        manager = PrefixManager()
        workspace = InMemoryWorkspace()
        runtime = StateGuardRuntime.create(worker=worker, manager=manager, workspace=workspace)
        harness = StateGuardHarness(
            runtime=runtime,
            flow_adapter=DACompWorkflow(5),
            config=StateGuardConfig(max_worker_steps=6),
        )
        result = harness.run(TaskSpec("dacomp-test", "query"))
        self.assertTrue(result.completed)
        self.assertEqual(result.final_answer, "answer")
        self.assertEqual(len(result.committed_states), 1)
        self.assertEqual(result.committed_states[0].source_step_start, 1)
        self.assertEqual(result.committed_states[0].source_step_end, 3)
        self.assertEqual(worker.injected, [])


if __name__ == "__main__":
    unittest.main()
