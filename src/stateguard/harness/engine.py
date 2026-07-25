from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from stateguard.adapters.flow import FlowAdapter, TurnFlowAdapter
from stateguard.agents.base import Agent
from stateguard.core.events import ManagerFailure, ReActStep
from stateguard.core.models import TaskSpec
from stateguard.repair.controller import RepairController, RepairDirective, RepairSession
from stateguard.runtime.checkpoints import CheckpointManager
from stateguard.runtime.bundle import StateGuardRuntime
from stateguard.runtime.executors import CodeExecutor
from stateguard.runtime.workspace import Workspace
from stateguard.state.draft import DraftStore, StateHeader, StateUpdate
from stateguard.state.graph import StateRelationGraph
from stateguard.state.models import AnalyticalState
from stateguard.state.store import StateStore
from stateguard.telemetry.artifacts import RunArtifactWriter
from stateguard.validation.models import ManagerAction, ManagerDecision

from .blind_view import BlindViewBuilder, ManagerObservation, assert_blind


class Manager(Protocol):
    def start_task(self, task: TaskSpec, state_index: list[dict[str, Any]]) -> None: ...

    def act(self, observation: ManagerObservation) -> ManagerDecision: ...


@dataclass(frozen=True)
class StateGuardConfig:
    max_worker_steps: int = 80
    max_manager_actions_per_event: int = 16
    max_repairs: int = 3
    light_repair_attempts: int = 2
    fail_open_on_abstain: bool = True

    @classmethod
    def from_json(cls, path: Path) -> "StateGuardConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class StateGuardResult:
    task_id: str
    final_answer: str
    completed: bool
    worker_steps: int
    manager_actions: int
    committed_states: tuple[AnalyticalState, ...]
    repair_count: int
    abstained_intervals: int
    runtime_id: str = ""
    degraded: bool = False
    manager_failures: tuple[ManagerFailure, ...] = ()


class StateGuardHarness:
    """Mechanism-only scheduler and executor for manager-selected actions."""

    def __init__(
        self,
        *,
        worker: Agent | None = None,
        manager: Manager | None = None,
        workspace: Workspace | None = None,
        flow_adapter: FlowAdapter | None = None,
        config: StateGuardConfig | None = None,
        state_store: StateStore | None = None,
        graph: StateRelationGraph | None = None,
        draft_store: DraftStore | None = None,
        repair_controller: RepairController | None = None,
        artifacts: RunArtifactWriter | None = None,
        checkpoints: CheckpointManager | None = None,
        worker_executor: CodeExecutor | None = None,
        runtime: StateGuardRuntime | None = None,
    ) -> None:
        self.flow_adapter = flow_adapter or TurnFlowAdapter()
        self.config = config or StateGuardConfig()
        if runtime is not None:
            supplied = {
                "worker": worker,
                "workspace": workspace,
                "state_store": state_store,
                "graph": graph,
                "draft_store": draft_store,
                "artifacts": artifacts,
                "checkpoints": checkpoints,
                "worker_executor": worker_executor,
            }
            conflicts = [name for name, value in supplied.items() if value is not None]
            if conflicts or manager is not None:
                raise ValueError(
                    "runtime is exclusive with separately supplied runtime components: "
                    f"{sorted(conflicts + (['manager'] if manager is not None else []))}"
                )
            runtime.assert_consistent()
            self.runtime = runtime
        else:
            if worker is None or workspace is None:
                raise ValueError("worker and workspace are required when runtime is not supplied")
            self.runtime = StateGuardRuntime.create(
                worker=worker,
                manager=manager,
                workspace=workspace,
                state_store=state_store,
                graph=graph,
                draft_store=draft_store,
                artifacts=artifacts,
                checkpoints=checkpoints,
                worker_executor=worker_executor,
            )
        self.worker = self.runtime.worker
        self.manager = self.runtime.manager
        self.workspace = self.runtime.workspace
        self.state_store = self.runtime.state_store
        self.graph = self.runtime.graph
        self.draft_store = self.runtime.draft_store
        self.trace_buffer = self.runtime.trace_buffer
        self.repair_controller = repair_controller or RepairController(
            self.config.max_repairs, self.config.light_repair_attempts
        )
        self.artifacts = self.runtime.artifacts
        self.checkpoints = self.runtime.checkpoints
        self.blind_view = BlindViewBuilder()
        self._manager_failures: list[ManagerFailure] = []
        self._manager_action_checkpoint = None
        self._manager_action_index: int | None = None

    def run(self, task: TaskSpec) -> StateGuardResult:
        self._manager_failures = []
        self._manager_action_index = None
        self.trace_buffer.start_unit(task.id)
        self.flow_adapter.start(task)
        staged_files = (
            self.workspace.stage_data_files(task.data_files) if task.data_files else {}
        )
        worker_prompt = task.initial_prompt()
        if staged_files:
            worker_prompt += (
                "\n\n<workspace_data_files>\n"
                + json.dumps(staged_files, ensure_ascii=False, indent=2)
                + "\n</workspace_data_files>\n"
                "Use the persistent python tool and the data_files mapping to read these files."
            )
        self.flow_adapter.prepare_worker(self.worker, worker_prompt)
        if self.manager is None:
            return self._run_without_manager(task)
        assert_blind(task.metadata)
        try:
            self.manager.start_task(task, self.state_store.index())
        except Exception as exc:
            self._record_manager_failure(
                phase="start_task", event_type="TASK_START", exc=exc
            )
            return self._run_without_manager(task)
        interval_start = self.checkpoints.capture("run:start")
        repair_session = RepairSession(interval_start)
        latest_step: ReActStep | None = None
        event_type = "TASK_START"
        total_worker_steps = 0
        total_manager_actions = 0
        total_repairs = 0
        abstained = 0

        while True:
            try:
                outcome, action_count, repair_delta, abstain_delta = self._manager_cycle(
                    task=task,
                    event_type=event_type,
                    latest_step=latest_step,
                    repair_session=repair_session,
                )
            except Exception as exc:
                if self._manager_action_checkpoint is not None:
                    self.checkpoints.restore(self._manager_action_checkpoint)
                self._record_manager_failure(
                    phase="act_or_execute",
                    event_type=event_type,
                    exc=exc,
                    action_index=self._manager_action_index,
                )
                total_manager_actions += 1
                if self.worker.done:
                    break
                outcome, action_count, repair_delta, abstain_delta = (
                    "RESUME",
                    0,
                    0,
                    0,
                )
            total_manager_actions += action_count
            total_repairs += repair_delta
            abstained += abstain_delta

            if outcome == "FINISH":
                break
            if self.worker.done:
                raise ValueError(
                    "manager attempted to resume a completed worker; COMMIT_STATE, repair, "
                    "or ABSTAIN is required"
                )
            steps_since_review = 0
            while True:
                if total_worker_steps >= self.config.max_worker_steps:
                    raise RuntimeError(
                        f"StateGuard exceeded max_worker_steps={self.config.max_worker_steps}"
                    )
                latest_step = self.worker.step()
                total_worker_steps += 1
                steps_since_review += 1
                self.trace_buffer.append(
                    latest_step, repair_attempt=repair_session.attempts
                )
                self.artifacts.record("worker", latest_step)
                if self.flow_adapter.should_review(latest_step, steps_since_review):
                    event_type = self.flow_adapter.review_event_type(latest_step)
                    break

        passed_step_ids = self.trace_buffer.pass_uncommitted()
        if passed_step_ids:
            self.artifacts.record(
                "trace",
                {
                    "event": "uncommitted_trace_passed",
                    "step_ids": list(passed_step_ids),
                },
            )
        if self.draft_store.current is not None:
            self.artifacts.record(
                "state",
                {
                    "event": "uncommitted_draft_discarded",
                    "state_id": self.draft_store.current.header.id,
                },
            )
            self.draft_store.discard()
        result = StateGuardResult(
            task_id=task.id,
            final_answer=self.worker.final_answer or "",
            completed=self.worker.done,
            worker_steps=total_worker_steps,
            manager_actions=total_manager_actions,
            committed_states=self.state_store.all(),
            repair_count=total_repairs,
            abstained_intervals=abstained,
            runtime_id=self.runtime.id,
            degraded=bool(self._manager_failures),
            manager_failures=tuple(self._manager_failures),
        )
        self.artifacts.write_summary(result)
        return result

    def _run_without_manager(self, task: TaskSpec) -> StateGuardResult:
        """Pure worker ReAct path: no StateGuard observation or state mutation."""
        total_worker_steps = 0
        while not self.worker.done:
            if total_worker_steps >= self.config.max_worker_steps:
                raise RuntimeError(
                    f"StateGuard exceeded max_worker_steps={self.config.max_worker_steps}"
                )
            step = self.worker.step()
            total_worker_steps += 1
            self.artifacts.record("worker", step)
        result = StateGuardResult(
            task_id=task.id,
            final_answer=self.worker.final_answer or "",
            completed=self.worker.done,
            worker_steps=total_worker_steps,
            manager_actions=0,
            committed_states=self.state_store.all(),
            repair_count=0,
            abstained_intervals=0,
            runtime_id=self.runtime.id,
            degraded=bool(self._manager_failures),
            manager_failures=tuple(self._manager_failures),
        )
        self.artifacts.write_summary(result)
        return result

    def _manager_cycle(
        self,
        *,
        task: TaskSpec,
        event_type: str,
        latest_step: ReActStep | None,
        repair_session: RepairSession,
    ) -> tuple[str, int, int, int]:
        if self.manager is None:
            raise RuntimeError("manager cycle is unavailable when manager=None")
        last_action_result: dict[str, Any] | None = None
        repairs = 0
        abstained = 0

        for action_index in range(1, self.config.max_manager_actions_per_event + 1):
            self._manager_action_index = action_index
            self._manager_action_checkpoint = self.checkpoints.capture(
                f"manager:before:{event_type}:{action_index}"
            )
            observation = self._manager_observation(
                task, event_type, latest_step, repair_session, last_action_result
            )
            command = self.manager.act(observation)
            self.artifacts.record(
                "manager",
                {"event_type": event_type, "observation": observation, "command": command},
            )

            if command.action is ManagerAction.RESUME_WORKER:
                if self.worker.done:
                    raise ValueError("RESUME_WORKER is illegal after a final worker answer")
                return "RESUME", action_index, repairs, abstained

            if command.action is ManagerAction.OPEN_STATE:
                header = self._open_state(command)
                provisional_ids = tuple(self._relation_ids(header.relations))
                hint_ids = self.flow_adapter.hint_state_ids(provisional_ids)
                hint = self._relation_hint(header.id, hint_ids)
                should_inject = bool(hint)
                if hint:
                    self.worker.inject_observation(hint)
                state_start = self.checkpoints.capture(f"draft:{header.id}:start")
                repair_session.begin_state(header.id, state_start)
                last_action_result = {
                    "action": "OPEN_STATE",
                    "state_id": header.id,
                    "relation_ids": list(provisional_ids),
                    "hint_state_ids": list(hint_ids),
                    "injected": should_inject,
                }
                event_type = "ACTION_RESULT"
                continue

            if command.action is ManagerAction.UPDATE_STATE:
                draft = self._update_state(command.state_update)
                last_action_result = {
                    "action": "UPDATE_STATE",
                    "state_id": draft.header.id,
                    "traced_step_ids": list(draft.traced_step_ids),
                }
                event_type = "ACTION_RESULT"
                continue

            if command.action is ManagerAction.FINALIZE_RELATIONS:
                draft = self._finalize_relations(command)
                last_action_result = {
                    "action": "FINALIZE_RELATIONS",
                    "state_id": draft.header.id,
                    "mode": command.relation_finalization.mode.value,
                    "relation_ids": self._relation_ids(draft.relations),
                    "changed": set(draft.relations) != set(draft.header.relations),
                }
                event_type = "ACTION_RESULT"
                continue

            if command.action is ManagerAction.COMMIT_STATE:
                if command.state_update is not None:
                    self._update_state(command.state_update)
                state = self._commit_draft()
                checkpoint = self.checkpoints.capture(f"state:{state.id}")
                repair_session.complete_state(checkpoint)
                last_action_result = {"action": "COMMIT_STATE", "state_id": state.id}
                if self.worker.done:
                    return "FINISH", action_index, repairs, abstained
                event_type = "ACTION_RESULT"
                continue

            if command.action is ManagerAction.REPAIR:
                draft = self.draft_store.current
                if draft is None:
                    raise ValueError("REPAIR requires an open current state")
                if repair_session.state_id != draft.header.id:
                    raise ValueError(
                        "repair budget is not bound to the current analytical state"
                    )
                current_branch = self.checkpoints.capture(
                    f"repair:original:attempt:{repair_session.attempts}"
                )
                rejected_step_ids = self.trace_buffer.reject_attempt(
                    repair_session.attempts
                )
                self.draft_store.reset_content_for_retry()
                directive = self.repair_controller.apply(
                    decision=command,
                    session=repair_session,
                    current_branch=current_branch,
                    worker=self.worker,
                    workspace=self.workspace,
                )
                self.artifacts.record("repair", repair_session.records[-1])
                self.artifacts.record(
                    "trace",
                    {
                        "event": "repair_branch_rejected",
                        "repair_attempt": repair_session.attempts - 1,
                        "rejected_step_ids": list(rejected_step_ids),
                    },
                )
                if directive is not RepairDirective.RETRY:
                    raise AssertionError("a scheduled REPAIR must produce a worker retry")
                repairs += 1
                return "RESUME", action_index, repairs, abstained

            if command.action is ManagerAction.ROLLBACK_PASS:
                if repair_session.attempts < self.repair_controller.max_repairs:
                    raise ValueError(
                        "ROLLBACK_PASS is legal only after the heavy retry was checked"
                    )
                self.repair_controller.rollback_pass(repair_session, self.checkpoints)
                self.artifacts.record("repair", repair_session.records[-1])
                abstained += 1
                if not self.config.fail_open_on_abstain:
                    raise RuntimeError("repair schedule exhausted")
                if self.worker.done:
                    return "FINISH", action_index, repairs, abstained
                return "RESUME", action_index, repairs, abstained

            if command.action is ManagerAction.ABSTAIN:
                current_branch = self.checkpoints.capture("manager:abstain:original")
                if repair_session.original_branch is None:
                    repair_session.original_branch = current_branch
                self.repair_controller.abstain(repair_session, self.checkpoints)
                self.artifacts.record("repair", repair_session.records[-1])
                abstained += 1
                if not self.config.fail_open_on_abstain:
                    raise RuntimeError("manager abstained")
                if self.worker.done:
                    return "FINISH", action_index, repairs, abstained
                return "RESUME", action_index, repairs, abstained

            raise AssertionError(f"unhandled manager action: {command.action}")

        self._manager_action_checkpoint = self.checkpoints.capture("manager:action_limit")
        raise RuntimeError(
            f"manager exceeded max_manager_actions_per_event={self.config.max_manager_actions_per_event}"
        )

    def _record_manager_failure(
        self,
        *,
        phase: str,
        event_type: str,
        exc: Exception,
        action_index: int | None = None,
    ) -> None:
        failure = ManagerFailure(
            phase=phase,
            event_type=event_type,
            error_type=type(exc).__name__,
            message=str(exc),
            action_index=action_index,
        )
        self._manager_failures.append(failure)
        self.artifacts.record("manager_failure", failure)

    def _manager_observation(
        self,
        task: TaskSpec,
        event_type: str,
        latest_step: ReActStep | None,
        repair_session: RepairSession,
        last_action_result: dict[str, Any] | None,
    ) -> ManagerObservation:
        draft = self.draft_store.current
        relation_states: list[dict[str, Any]] = []
        if draft is not None:
            relation_states = [
                self.state_store.get(state_id).to_dict()
                for state_id in self._relation_ids(draft.relations)
            ]
        return self.blind_view.build(
            task=task,
            event_type=event_type,
            flow_policy=self.flow_adapter.manager_context(),
            available_state_id=f"S{len(self.state_store) + 1}",
            worker_step=latest_step,
            untraced_steps=tuple(self.trace_buffer.steps),
            trace_history=self.trace_buffer.history(),
            current_draft=draft.to_dict() if draft else None,
            state_index=self.state_store.index(),
            stored_states=self.state_store.relation_catalog(),
            relation_states=relation_states,
            workspace_manifest=self.workspace.manifest(),
            repair_attempts=repair_session.attempts,
            last_action_result=last_action_result,
        )

    def _open_state(self, command: ManagerDecision) -> StateHeader:
        header = command.state_header
        if header is None:
            raise ValueError("OPEN_STATE omitted state_header")
        available = f"S{len(self.state_store) + 1}"
        if header.id != available:
            raise ValueError(f"manager must write available state id {available}, got {header.id}")
        self.flow_adapter.validate_state_open(header, tuple(self.trace_buffer.steps))
        for state_id in self._relation_ids(header.relations):
            self.state_store.get(state_id)
        self.draft_store.open(header)
        return header

    def _update_state(self, update: StateUpdate | None):
        if update is None:
            raise ValueError("state update is missing")
        draft = self.draft_store.current
        if draft is not None and draft.relations_finalized:
            raise ValueError(
                "cannot UPDATE_STATE after FINALIZE_RELATIONS; write current state first"
            )
        self.trace_buffer.consume(update.traced_step_ids)
        return self.draft_store.update(update)

    def _finalize_relations(self, command: ManagerDecision):
        finalization = command.relation_finalization
        if finalization is None:
            raise ValueError("FINALIZE_RELATIONS omitted relation_finalization")
        draft = self.draft_store.current
        if draft is None:
            raise ValueError("manager must OPEN_STATE before FINALIZE_RELATIONS")
        self.flow_adapter.validate_relation_finalization(
            draft=draft,
            finalization=finalization,
            untraced_steps=tuple(self.trace_buffer.steps),
        )
        for state_id in self._relation_ids(finalization.relations):
            self.state_store.get(state_id)
        return self.draft_store.finalize_relations(finalization)

    def _commit_draft(self) -> AnalyticalState:
        transaction = self.checkpoints.capture("state:precommit")
        try:
            draft = self.draft_store.current
            if draft is None:
                raise ValueError("manager must OPEN_STATE before COMMIT_STATE")
            traced_step_ids = tuple(draft.traced_step_ids)
            state = self.draft_store.close()
            self.state_store.commit(state)
            self.graph.add_state(state)
            self.trace_buffer.accept(traced_step_ids)
            self.artifacts.record(
                "trace",
                {"event": "state_committed", "history": self.trace_buffer.history()},
            )
        except Exception:
            self.checkpoints.restore(transaction)
            raise
        return state

    def _relation_hint(self, state_id: str, hint_state_ids: tuple[str, ...]) -> str:
        if not hint_state_ids:
            return ""
        states = [self.state_store.get(item).as_state_hint() for item in hint_state_ids]
        payload = {
            "new_state_id": state_id,
            "relation_state_ids": list(hint_state_ids),
            "states": states,
        }
        return (
            "<analytical_state_hint>\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            + "\n</analytical_state_hint>\n"
            "These manager-selected states are observations. Verify them before reuse."
        )

    @staticmethod
    def _relation_ids(relations) -> list[str]:
        return [
            relation.related_state_id
            for relation in relations
            if relation.related_state_id is not None
        ]
