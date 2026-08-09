from __future__ import annotations

import copy
import unittest

from stateguard.adapters.dabstep.workflow import DABstepWorkflow
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


class SevenStepWorker:
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
        self._messages.append(Message("user", prompt.query))

    def continue_turn(self, prompt):
        raise AssertionError(prompt)

    def step(self):
        self.index += 1
        terminal = self.index == 7
        if terminal:
            self._done = True
            self._answer = "answer"
            action = AgentAction(kind="final", answer="answer")
        else:
            action = AgentAction(
                kind="tool", tool_name="python_interpreter", arguments={"code": "x=1"}
            )
        observation = ToolResult(
            "python_interpreter",
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
        return copy.deepcopy(
            (self._messages, self.index, self._done, self._answer, self.injected)
        )

    def restore(self, snapshot):
        (
            self._messages,
            self.index,
            self._done,
            self._answer,
            self.injected,
        ) = copy.deepcopy(snapshot)


class IntervalManager:
    def __init__(self):
        self.calls = 0

    def configure_lifecycle(self, prompt):
        self.lifecycle = prompt

    def start_task(self, task):
        self.task = task

    def act(self, observation):
        self.calls += 1
        if self.calls == 1:
            self._assert_ids(observation, (1, 2, 3))
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    "S1", (Constraint("answer the query"),), ()
                ),
            )
        if self.calls == 2:
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    issue="important result formed",
                    conclusions=(Conclusion("steps 1-2 establish the result"),),
                    traced_step_ids=(1, 2),
                ),
            )
        if self.calls == 3:
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    RelationFinalizationMode.SELECT,
                    (StateRelation(StateRelationType.INIT),),
                    "no predecessor",
                ),
            )
        if self.calls == 4:
            return ManagerDecision(action=ManagerAction.COMMIT_STATE)
        if self.calls in {5, 6}:
            return ManagerDecision(action=ManagerAction.RESUME_WORKER)
        self._assert_ids(observation, (3, 4, 5, 6, 7))
        return ManagerDecision(action=ManagerAction.ABSTAIN)

    @staticmethod
    def _assert_ids(observation, expected):
        assert tuple(step.step_id for step in observation.untraced_steps) == expected


class DABstepManagedFlowTests(unittest.TestCase):
    def test_three_step_pause_allows_two_step_state_and_no_state_hint(self):
        worker = SevenStepWorker()
        manager = IntervalManager()
        runtime = StateGuardRuntime.create(
            worker=worker, manager=manager, workspace=InMemoryWorkspace()
        )
        result = StateGuardHarness(
            runtime=runtime,
            flow_adapter=DABstepWorkflow(3),
            config=StateGuardConfig(max_worker_steps=7),
        ).run(TaskSpec("dabstep-test", "query"))
        self.assertTrue(result.completed)
        self.assertEqual(result.final_answer, "answer")
        self.assertEqual(len(result.committed_states), 1)
        self.assertEqual(result.committed_states[0].source_step_start, 1)
        self.assertEqual(result.committed_states[0].source_step_end, 2)
        self.assertEqual(worker.injected, [])


if __name__ == "__main__":
    unittest.main()
