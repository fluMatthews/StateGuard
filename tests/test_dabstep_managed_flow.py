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
    SourceInterval,
    StateHeader,
    StateUpdate,
)
from stateguard.state.models import Conclusion, Constraint, StateRelation, StateRelationType
from stateguard.validation.models import (
    ERROR_HINT_PROMPT,
    AnalyticalEvidence,
    ErrorHint,
    ManagerAction,
    ManagerDecision,
)


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
            assert observation.committed_state_index is None
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
                    source_interval=SourceInterval(1, 2),
                ),
            )
        if self.calls == 3:
            assert observation.committed_state_index == ()
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
        if self.calls == 7:
            self._assert_ids(observation, (3, 4, 5, 6, 7))
            assert observation.committed_state_index is None
            assert observation.flow_policy["terminal_pending_review"]
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    "S2", (Constraint("answer the query"),), ()
                ),
            )
        if self.calls == 8:
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    issue="terminal answer and supporting tail",
                    conclusions=(Conclusion("steps 3-7 complete the answer"),),
                    source_interval=SourceInterval(3, 7),
                ),
            )
        if self.calls == 9:
            assert tuple(
                item["id"] for item in observation.committed_state_index
            ) == ("S1",)
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    RelationFinalizationMode.SELECT,
                    (StateRelation(StateRelationType.PROGRESS, "S1"),),
                    "terminal result follows the earlier analysis",
                ),
            )
        if self.calls == 10:
            return ManagerDecision(action=ManagerAction.COMMIT_STATE)
        raise AssertionError(f"unexpected Manager call {self.calls}")

    @staticmethod
    def _assert_ids(observation, expected):
        assert tuple(step.step_id for step in observation.untraced_steps) == expected




class RepairTailWorker:
    """One terminal step initially; each repair adds two tools and a new final."""

    def __init__(self, retry_length=3):
        self._messages = []
        self.index = 0
        self.generation = 0
        self.retry_length = retry_length
        self.retry_steps_remaining = 0
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
        if self.generation == 0:
            terminal = True
        else:
            self.retry_steps_remaining -= 1
            terminal = self.retry_steps_remaining == 0
        if terminal:
            self._done = True
            self._answer = f"answer-{self.generation}"
            action = AgentAction(kind="final", answer=self._answer)
        else:
            action = AgentAction(
                kind="tool",
                tool_name="python_interpreter",
                arguments={"code": f"value={self.index}"},
            )
        observation = ToolResult(
            "python_interpreter",
            True,
            f"step-{self.index}",
            {"execution_attempted": True, "execution_succeeded": True},
        )
        return ReActStep(self.index, action, observation, terminal)

    def inject_observation(self, content, metadata=None):
        self.injected.append((content, metadata))
        self._messages.append(Message("user", str(content)))
        self.generation += 1
        self.retry_steps_remaining = self.retry_length
        self._done = False
        self._answer = None

    def snapshot(self):
        return copy.deepcopy(self.__dict__)

    def restore(self, snapshot):
        self.__dict__ = copy.deepcopy(snapshot)


class TerminalRepairManager:
    def __init__(self):
        self.observations = []

    def configure_lifecycle(self, prompt):
        self.lifecycle = prompt

    def start_task(self, task):
        self.task = task

    def act(self, observation):
        self.observations.append(observation)
        draft = observation.current_draft
        if draft is None:
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    observation.available_state_id,
                    (Constraint("answer the query"),),
                    (),
                ),
            )
        if not draft["conclusions"]:
            pending = tuple(step.step_id for step in observation.untraced_steps)
            start = (draft.get("source_interval") or {"start": pending[0]})["start"]
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    issue=f"terminal result for {draft['id']}",
                    conclusions=(Conclusion(f"accepted result for {draft['id']}"),),
                    source_interval=SourceInterval(start, pending[-1]),
                ),
            )
        if observation.repair_attempts == 0:
            return _repair_decision(
                draft["id"],
                tuple(range(draft["source_interval"]["start"], draft["source_interval"]["end"] + 1)),
            )
        if not draft["relations_finalized"]:
            relations = (StateRelation(StateRelationType.INIT),)
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    RelationFinalizationMode.SELECT,
                    relations,
                    "posthoc relation selection",
                ),
            )
        return ManagerDecision(action=ManagerAction.COMMIT_STATE)


class BudgetExhaustedManager:
    def __init__(self):
        self.repair_rejected = False

    def configure_lifecycle(self, prompt):
        self.lifecycle = prompt

    def start_task(self, task):
        self.task = task

    def act(self, observation):
        if observation.last_action_result and (
            observation.last_action_result.get("action") == "ACTION_REJECTED"
        ):
            self.repair_rejected = True
            assert "budget is exhausted" in observation.last_action_result["error"]
            return ManagerDecision(action=ManagerAction.ABSTAIN)
        draft = observation.current_draft
        if draft is None:
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    observation.available_state_id,
                    (Constraint("answer the query"),),
                    (),
                ),
            )
        if not draft["conclusions"]:
            pending = tuple(step.step_id for step in observation.untraced_steps)
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    issue="terminal result",
                    conclusions=(Conclusion("unsupported terminal result"),),
                    source_interval=SourceInterval(pending[0], pending[-1]),
                ),
            )
        return _repair_decision(
            draft["id"],
            tuple(range(draft["source_interval"]["start"], draft["source_interval"]["end"] + 1)),
        )


def _repair_decision(state_id, step_ids):
    return ManagerDecision(
        action=ManagerAction.REPAIR,
        evidence=AnalyticalEvidence(
            violated_constraints=("The terminal result conflicts with its evidence.",),
            evidence=("The current terminal trace contains a concrete mismatch.",),
            suspected_state_ids=(state_id,),
            suspected_step_ids=step_ids,
        ),
        error_hint=ErrorHint(
            ERROR_HINT_PROMPT,
            ("terminal_result",),
            "The terminal result is inconsistent with the evidence recorded in this state.",
        ),
    )


class DanglingDraftManager:
    """Leaves a draft open when the Worker finishes -- the real failure shape.

    The Manager cannot OPEN_STATE for the tail because one is already open, so
    a terminal review that waits for an OPEN_STATE never arrives and the tail
    is passed unreviewed. The harness must adopt the open draft instead.
    """

    def __init__(self):
        self.calls = 0
        self.saw_terminal_review = False
        self.terminal_remaining = None

    def configure_lifecycle(self, prompt):
        self.lifecycle = prompt

    def start_task(self, task):
        self.task = task

    def act(self, observation):
        self.calls += 1
        if self.calls == 1:
            assert not observation.flow_policy["terminal_pending_review"]
            return ManagerDecision(
                action=ManagerAction.OPEN_STATE,
                state_header=StateHeader(
                    "S1", (Constraint("answer the query"),), ()
                ),
            )
        if self.calls in {2, 3}:
            # Mid-trajectory with a draft open: still not a terminal review.
            assert not observation.flow_policy["terminal_pending_review"]
            return ManagerDecision(action=ManagerAction.RESUME_WORKER)
        if self.calls == 4:
            self.saw_terminal_review = observation.flow_policy[
                "terminal_pending_review"
            ]
            self.terminal_remaining = observation.flow_policy[
                "worker_steps_remaining"
            ]
            return ManagerDecision(
                action=ManagerAction.UPDATE_STATE,
                state_update=StateUpdate(
                    issue="the whole trajectory answers the query",
                    conclusions=(Conclusion("steps 1-7 produce the answer"),),
                    source_interval=SourceInterval(1, 7),
                ),
            )
        if self.calls == 5:
            return ManagerDecision(
                action=ManagerAction.FINALIZE_RELATIONS,
                relation_finalization=RelationFinalization(
                    RelationFinalizationMode.SELECT,
                    (StateRelation(StateRelationType.INIT),),
                    "no predecessor",
                ),
            )
        if self.calls == 6:
            return ManagerDecision(action=ManagerAction.COMMIT_STATE)
        raise AssertionError(f"unexpected Manager call {self.calls}")


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
        self.assertFalse(result.degraded)
        self.assertEqual(len(result.committed_states), 2)
        self.assertEqual(result.committed_states[0].source_step_start, 1)
        self.assertEqual(result.committed_states[0].source_step_end, 2)
        self.assertEqual(result.committed_states[1].source_step_start, 3)
        self.assertEqual(result.committed_states[1].source_step_end, 7)
        # A single-query flow injects no relation state_hint: the Manager picks
        # its own state boundaries, so there is nothing to hint at OPEN_STATE.
        # It does receive the resume state_summary once the store grows.
        kinds = [(metadata or {}).get("stateguard") for _, metadata in worker.injected]
        self.assertEqual([k for k in kinds if k == "state_hint"], [])
        self.assertEqual(kinds, ["state_summary"])

    def test_open_draft_at_worker_finish_is_adopted_as_the_terminal_state(self):
        worker = SevenStepWorker()
        manager = DanglingDraftManager()
        runtime = StateGuardRuntime.create(
            worker=worker, manager=manager, workspace=InMemoryWorkspace()
        )
        result = StateGuardHarness(
            runtime=runtime,
            flow_adapter=DABstepWorkflow(3),
            config=StateGuardConfig(max_worker_steps=10),
        ).run(TaskSpec("dangling-draft", "query"))

        self.assertTrue(manager.saw_terminal_review)
        self.assertEqual(manager.terminal_remaining, 3)
        self.assertTrue(result.completed)
        self.assertFalse(result.degraded)
        self.assertEqual([state.id for state in result.committed_states], ["S1"])
        self.assertEqual(result.committed_states[0].source_step_start, 1)
        self.assertEqual(result.committed_states[0].source_step_end, 7)
        # Nothing is left to the silent end-of-run pass: every step was reviewed.
        statuses = {
            item["step_id"]: item["status"] for item in runtime.trace_buffer.history()
        }
        self.assertEqual(set(statuses.values()), {"accepted"})

    def test_terminal_state_is_checked_once_then_worker_runs_to_completion(self):
        worker = RepairTailWorker(retry_length=5)
        manager = TerminalRepairManager()
        runtime = StateGuardRuntime.create(
            worker=worker, manager=manager, workspace=InMemoryWorkspace()
        )
        result = StateGuardHarness(
            runtime=runtime,
            flow_adapter=DABstepWorkflow(3),
            config=StateGuardConfig(max_worker_steps=10),
        ).run(TaskSpec("terminal-groups", "query"))

        self.assertTrue(result.completed)
        self.assertFalse(result.degraded)
        self.assertEqual(result.worker_steps, 6)
        self.assertEqual(result.repair_count, 1)
        self.assertEqual([state.id for state in result.committed_states], ["S1"])
        # The terminal repair keeps the original start and extends the source
        # interval through the retry steps observed at the next review.
        self.assertEqual(result.committed_states[0].source_step_end, 4)
        self.assertEqual(result.abstained_intervals, 0)
        statuses = {
            item["step_id"]: item["status"] for item in runtime.trace_buffer.history()
        }
        self.assertEqual(statuses[3], "accepted")
        self.assertEqual(statuses[6], "passed")

    def test_terminal_error_stops_cleanly_when_worker_budget_is_exhausted(self):
        worker = RepairTailWorker()
        manager = BudgetExhaustedManager()
        runtime = StateGuardRuntime.create(
            worker=worker, manager=manager, workspace=InMemoryWorkspace()
        )
        result = StateGuardHarness(
            runtime=runtime,
            flow_adapter=DABstepWorkflow(3),
            config=StateGuardConfig(max_worker_steps=1),
        ).run(TaskSpec("terminal-budget", "query"))

        self.assertTrue(result.completed)
        self.assertFalse(result.degraded)
        self.assertTrue(manager.repair_rejected)
        self.assertEqual(result.worker_steps, 1)
        self.assertEqual(result.repair_count, 0)
        self.assertEqual(result.committed_states, ())
        self.assertEqual(result.abstained_intervals, 1)


if __name__ == "__main__":
    unittest.main()
