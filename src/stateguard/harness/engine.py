from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

from stateguard.adapters.flow import FlowAdapter, TurnFlowAdapter
from stateguard.agents.base import Agent
from stateguard.core.events import ManagerFailure, ReActStep
from stateguard.core.models import TaskSpec
from stateguard.repair.controller import RepairController, RepairDirective, RepairSession
from stateguard.runtime.checkpoints import CheckpointManager, CheckpointRef
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
    def start_task(self, task: TaskSpec) -> None: ...

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
        self._manager_action_index: int | None = None
        committed_numbers = [int(state.id[1:]) for state in self.state_store.all()]
        self._next_state_number = max(committed_numbers, default=0) + 1

    def run(self, task: TaskSpec) -> StateGuardResult:
        self._manager_failures = []
        self._manager_action_index = None
        self.trace_buffer.start_unit(task.id)
        self.flow_adapter.start(task)
        self.flow_adapter.prepare_worker(self.worker, task, self.workspace)
        if self.manager is None:
            return self._run_without_manager(task)
        assert_blind(task.metadata)
        try:
            configure_lifecycle = getattr(self.manager, "configure_lifecycle", None)
            if callable(configure_lifecycle):
                configure_lifecycle(self.flow_adapter.lifecycle_prompt())
            self.manager.start_task(task)
        except Exception as exc:
            self._record_manager_failure(
                phase="start_task", event_type="TASK_START", exc=exc
            )
            return self._run_without_manager(task)
        interval_start = self.checkpoints.capture("run:start")
        repair_session = RepairSession(interval_start)
        try:
            return self._run_managed_task(task, repair_session)
        finally:
            # A RepairSession never crosses a benchmark task unit/turn.  LongDS
            # reuses the live Worker/workspace/store, not historical snapshots.
            self._release_checkpoint_refs(
                interval_start,
                repair_session.interval_start,
                repair_session.original_branch,
            )

    def _run_managed_task(
        self,
        task: TaskSpec,
        repair_session: RepairSession,
    ) -> StateGuardResult:
        latest_step: ReActStep | None = None
        event_type = "TASK_START"
        total_worker_steps = 0
        total_manager_actions = 0
        total_repairs = 0
        abstained = 0
        manager_cycle_due = self.flow_adapter.review_before_worker()

        while True:
            if manager_cycle_due:
                try:
                    outcome, action_count, repair_delta, abstain_delta = self._manager_cycle(
                        task=task,
                        event_type=event_type,
                        latest_step=latest_step,
                        repair_session=repair_session,
                    )
                except Exception as exc:
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
            else:
                # Single-query workflows initialize the Manager prompt now but
                # request no control action until the first native review pause.
                outcome, action_count, repair_delta, abstain_delta = (
                    "RESUME",
                    0,
                    0,
                    0,
                )
                manager_cycle_due = True

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
        preflight_retry_used = False

        for action_index in range(1, self.config.max_manager_actions_per_event + 1):
            self._manager_action_index = action_index
            observation = self._manager_observation(
                task, event_type, latest_step, repair_session, last_action_result
            )
            command = self.manager.act(observation)
            self.artifacts.record(
                "manager",
                {"event_type": event_type, "observation": observation, "command": command},
            )
            try:
                self._preflight_action(command, repair_session)
            except (KeyError, TypeError, ValueError) as exc:
                if preflight_retry_used:
                    raise
                preflight_retry_used = True
                last_action_result = {
                    "action": "ACTION_REJECTED",
                    "requested_action": command.action.value,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retry_allowed": True,
                }
                self.artifacts.record(
                    "manager",
                    {
                        "event_type": event_type,
                        "event": "action_preflight_rejected",
                        **last_action_result,
                    },
                )
                event_type = "ACTION_RESULT"
                continue
            preflight_retry_used = False

            if command.action is ManagerAction.RESUME_WORKER:
                return "RESUME", action_index, repairs, abstained

            if command.action is ManagerAction.OPEN_STATE:
                header = command.state_header
                assert header is not None
                previous_interval = repair_session.interval_start
                provisional_ids = tuple(self._relation_ids(header.relations))
                hint_ids = self.flow_adapter.hint_state_ids(provisional_ids)
                hint = self._relation_hint(hint_ids)
                should_inject = bool(hint)
                with self._action_transaction(
                    f"open_state:{header.id}",
                    components=("worker", "state_draft"),
                    repair_session=repair_session,
                ):
                    self._open_state(command)
                    if hint:
                        self.worker.inject_observation(
                            hint, metadata={"stateguard": "state_hint"}
                        )
                    state_start = self.checkpoints.capture(f"draft:{header.id}:start")
                    repair_session.begin_state(header.id, state_start)
                self._next_state_number += 1
                self._release_checkpoint_refs(previous_interval)
                if hint:
                    self.artifacts.record(
                        "hint",
                        {
                            "kind": "state_hint",
                            "state_id": header.id,
                            "related_state_ids": list(hint_ids),
                            "content": hint,
                        },
                    )
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
                with self._action_transaction(
                    "update_state",
                    components=("state_draft", "trace_buffer"),
                ):
                    draft = self._update_state(
                        command.state_update,
                        allow_sparse_trace=repair_session.attempts > 0,
                    )
                last_action_result = {
                    "action": "UPDATE_STATE",
                    "state_id": draft.header.id,
                    "traced_step_ids": list(draft.traced_step_ids),
                }
                event_type = "ACTION_RESULT"
                continue

            if command.action is ManagerAction.FINALIZE_RELATIONS:
                with self._action_transaction(
                    "finalize_relations",
                    components=("state_draft",),
                ):
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
                retired_checkpoints = (
                    repair_session.interval_start,
                    repair_session.original_branch,
                )
                with self._action_transaction(
                    "commit_state",
                    components=(
                        "state_draft",
                        "state_store",
                        "state_graph",
                        "trace_buffer",
                    ),
                    repair_session=repair_session,
                ):
                    if command.state_update is not None:
                        self._update_state(
                            command.state_update,
                            allow_sparse_trace=repair_session.attempts > 0,
                        )
                    state = self._commit_draft()
                    checkpoint = self.checkpoints.capture(f"state:{state.id}")
                    repair_session.complete_state(checkpoint)
                self._release_checkpoint_refs(*retired_checkpoints)
                self.artifacts.record(
                    "trace",
                    {"event": "state_committed", "history": self.trace_buffer.history()},
                )
                last_action_result = {"action": "COMMIT_STATE", "state_id": state.id}
                if self.worker.done and not self.trace_buffer.steps:
                    return "FINISH", action_index, repairs, abstained
                event_type = "ACTION_RESULT"
                continue

            if command.action is ManagerAction.REPAIR:
                draft = self.draft_store.current
                assert draft is not None
                state_id = draft.header.id
                use_heavy = (
                    repair_session.attempts + 1
                    > self.repair_controller.light_repair_attempts
                )
                components = ["worker", "state_draft", "trace_buffer"]
                if use_heavy:
                    components.append("workspace")
                created_original = None
                try:
                    with self._action_transaction(
                        f"{'heavy' if use_heavy else 'light'}_repair:{state_id}",
                        components=tuple(components),
                        repair_session=repair_session,
                    ):
                        current_branch = repair_session.original_branch
                        if current_branch is None:
                            created_original = self.checkpoints.capture(
                                f"repair:original:{state_id}"
                            )
                            current_branch = created_original
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
                except Exception:
                    # If the first repair action itself was not applied atomically,
                    # its newly captured original branch is not owned by the restored
                    # RepairSession and must not remain retained.
                    self._release_checkpoint_refs(created_original)
                    raise
                self.artifacts.record("repair", repair_session.records[-1])
                self.artifacts.record(
                    "hint",
                    {
                        "kind": "error_hint",
                        "state_id": state_id,
                        "repair_attempt": repair_session.attempts,
                        "mode": repair_session.records[-1]["mode"],
                        "error_hint": command.error_hint,
                        "content": command.error_hint.as_observation(),
                    },
                )
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

            if command.action is ManagerAction.ABANDON_STATE:
                state_id = repair_session.state_id
                self.repair_controller.abandon_state(repair_session, self.checkpoints)
                repair_record = repair_session.records[-1]
                passed_step_ids = self._settle_abandoned_state(repair_session)
                self.artifacts.record("repair", repair_record)
                self.artifacts.record(
                    "trace",
                    {
                        "event": "state_abandoned",
                        "state_id": state_id,
                        "step_ids": list(passed_step_ids),
                    },
                )
                abstained += 1
                if not self.config.fail_open_on_abstain:
                    raise RuntimeError("repair schedule exhausted")
                if self.worker.done:
                    return "FINISH", action_index, repairs, abstained
                return "RESUME", action_index, repairs, abstained

            if command.action is ManagerAction.ABSTAIN:
                if repair_session.original_branch is not None:
                    self.repair_controller.abstain(repair_session, self.checkpoints)
                    self._settle_restored_repair_branch(repair_session)
                    self.artifacts.record("repair", repair_session.records[-1])
                else:
                    previous_interval = repair_session.interval_start
                    with self._action_transaction(
                        "abstain_pass",
                        components=("state_draft", "trace_buffer"),
                        repair_session=repair_session,
                    ):
                        passed_step_ids = self.trace_buffer.pass_uncommitted()
                        self.draft_store.discard()
                        clean = self.checkpoints.capture("manager:abstain:pass")
                        repair_session.complete_state(clean)
                    self._release_checkpoint_refs(previous_interval)
                    self.artifacts.record(
                        "trace",
                        {
                            "event": "manager_abstain_pass",
                            "step_ids": list(passed_step_ids),
                        },
                    )
                abstained += 1
                if not self.config.fail_open_on_abstain:
                    raise RuntimeError("manager abstained")
                if self.worker.done:
                    return "FINISH", action_index, repairs, abstained
                return "RESUME", action_index, repairs, abstained

            raise AssertionError(f"unhandled manager action: {command.action}")

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
        return self.blind_view.build(
            task=task,
            event_type=event_type,
            flow_policy=self.flow_adapter.manager_context(),
            available_state_id=f"S{self._next_state_number}",
            worker_step=latest_step,
            untraced_steps=tuple(self.trace_buffer.steps),
            current_draft=draft.to_dict() if draft else None,
            workspace_manifest=self.workspace.manifest(),
            repair_attempts=repair_session.attempts,
            last_action_result=last_action_result,
        )

    def _preflight_action(
        self,
        command: ManagerDecision,
        repair_session: RepairSession,
    ) -> None:
        """Reject only definite protocol/lifecycle errors before any mutation."""
        if command.action is ManagerAction.RESUME_WORKER:
            if self.worker.done:
                raise ValueError("RESUME_WORKER is illegal after a final worker answer")
            return

        if command.action is ManagerAction.OPEN_STATE:
            header = command.state_header
            if header is None:
                raise ValueError("OPEN_STATE omitted state_header")
            available = f"S{self._next_state_number}"
            if header.id != available:
                raise ValueError(
                    f"manager must write available state id {available}, got {header.id}"
                )
            if self.draft_store.current is not None:
                raise ValueError(
                    f"state {self.draft_store.current.header.id} is still open"
                )
            self.flow_adapter.validate_state_open(header, tuple(self.trace_buffer.steps))
            self._validate_relation_ids(header.relations)
            return

        if command.action is ManagerAction.UPDATE_STATE:
            self._preflight_update(
                command.state_update,
                allow_sparse_trace=repair_session.attempts > 0,
            )
            return

        if command.action is ManagerAction.FINALIZE_RELATIONS:
            finalization = command.relation_finalization
            draft = self.draft_store.current
            if finalization is None:
                raise ValueError("FINALIZE_RELATIONS omitted relation_finalization")
            if draft is None:
                raise ValueError("manager must OPEN_STATE before FINALIZE_RELATIONS")
            self.flow_adapter.validate_relation_finalization(
                draft=draft,
                finalization=finalization,
                untraced_steps=tuple(self.trace_buffer.steps),
            )
            self._validate_relation_ids(finalization.relations)
            return

        if command.action is ManagerAction.COMMIT_STATE:
            draft = self.draft_store.current
            if draft is None:
                raise ValueError("manager must OPEN_STATE before COMMIT_STATE")
            if command.state_update is not None:
                self._preflight_update(
                    command.state_update,
                    allow_sparse_trace=repair_session.attempts > 0,
                )
            if not draft.relations_finalized:
                raise ValueError("manager must FINALIZE_RELATIONS before COMMIT_STATE")
            if any(state.id == draft.header.id for state in self.state_store.all()):
                raise ValueError(f"state already committed: {draft.header.id}")
            self._validate_relation_ids(draft.relations)
            return

        if command.action is ManagerAction.REPAIR:
            draft = self.draft_store.current
            if draft is None:
                raise ValueError("REPAIR requires an open current state")
            if repair_session.state_id != draft.header.id:
                raise ValueError(
                    "repair budget is not bound to the current analytical state"
                )
            if repair_session.attempts >= self.repair_controller.max_repairs:
                raise ValueError("repair schedule exhausted; manager must ABANDON_STATE")
            return

        if command.action is ManagerAction.ABANDON_STATE:
            if repair_session.attempts < self.repair_controller.max_repairs:
                raise ValueError(
                    "ABANDON_STATE is legal only after the heavy retry was checked"
                )
            if repair_session.original_branch is None:
                raise ValueError("ABANDON_STATE requires an active repair chain")
            return

        if command.action is ManagerAction.ABSTAIN:
            return

        raise ValueError(f"unsupported manager action: {command.action}")

    def _preflight_update(
        self,
        update: StateUpdate | None,
        *,
        allow_sparse_trace: bool,
    ) -> None:
        if update is None:
            raise ValueError("state update is missing")
        draft = self.draft_store.snapshot()
        if draft is None:
            raise ValueError("manager must OPEN_STATE before UPDATE_STATE")
        if draft.relations_finalized:
            raise ValueError(
                "cannot UPDATE_STATE after FINALIZE_RELATIONS; write current state first"
            )
        self.trace_buffer.validate_selection(
            update.traced_step_ids,
            allow_sparse=allow_sparse_trace,
        )
        # Apply to the detached draft to reuse state-local validation without
        # changing the live draft before the action transaction begins.
        draft.apply(update)

    def _validate_relation_ids(self, relations: Any) -> None:
        for state_id in self._relation_ids(relations):
            self.state_store.get(state_id)

    def _release_checkpoint_refs(
        self, *references: CheckpointRef | None
    ) -> None:
        """Release dead lifecycle checkpoints once no RepairSession can use them."""
        released: set[str] = set()
        for reference in references:
            if reference is None or reference.id in released:
                continue
            released.add(reference.id)
            if self.checkpoints.is_retained(reference):
                self.checkpoints.release(reference)

    def _settle_restored_repair_branch(self, repair_session: RepairSession) -> None:
        """Keep only the restored branch and permanently exhaust this state repair."""
        restored = repair_session.original_branch
        if restored is None:
            raise ValueError("restored repair branch is missing")
        previous_interval = repair_session.interval_start
        repair_session.interval_start = restored
        repair_session.original_branch = None
        repair_session.attempts = self.repair_controller.max_repairs
        self._release_checkpoint_refs(previous_interval)


    def _settle_abandoned_state(
        self, repair_session: RepairSession
    ) -> tuple[int, ...]:
        """Discard the failed draft after restoring the original Worker branch."""
        restored = repair_session.original_branch
        if restored is None:
            raise ValueError("restored repair branch is missing")
        previous_interval = repair_session.interval_start
        passed_step_ids = self.trace_buffer.pass_uncommitted()
        self.draft_store.discard()
        clean = self.checkpoints.capture("state:abandoned")
        repair_session.complete_state(clean)
        self._release_checkpoint_refs(previous_interval, restored)
        return passed_step_ids
    @contextmanager
    def _action_transaction(
        self,
        label: str,
        *,
        components: tuple[str, ...],
        repair_session: RepairSession | None = None,
    ) -> Iterator[None]:
        """Rollback only components the current Manager action can mutate."""
        checkpoint = self.checkpoints.capture(
            f"manager_action:{label}", components=components
        )
        repair_snapshot = repair_session.snapshot() if repair_session is not None else None
        try:
            yield
        except Exception:
            try:
                self.checkpoints.restore(checkpoint)
            finally:
                if repair_session is not None and repair_snapshot is not None:
                    repair_session.restore(repair_snapshot)
                self.checkpoints.release(checkpoint)
            raise
        else:
            self.checkpoints.release(checkpoint)

    def _open_state(self, command: ManagerDecision) -> StateHeader:
        header = command.state_header
        if header is None:
            raise ValueError("OPEN_STATE omitted state_header")
        available = f"S{self._next_state_number}"
        if header.id != available:
            raise ValueError(f"manager must write available state id {available}, got {header.id}")
        self.flow_adapter.validate_state_open(header, tuple(self.trace_buffer.steps))
        for state_id in self._relation_ids(header.relations):
            self.state_store.get(state_id)
        self.draft_store.open(header)
        return header

    def _update_state(
        self,
        update: StateUpdate | None,
        *,
        allow_sparse_trace: bool,
    ):
        if update is None:
            raise ValueError("state update is missing")
        draft = self.draft_store.current
        if draft is not None and draft.relations_finalized:
            raise ValueError(
                "cannot UPDATE_STATE after FINALIZE_RELATIONS; write current state first"
            )
        self.trace_buffer.consume(
            update.traced_step_ids,
            allow_sparse=allow_sparse_trace,
        )
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
        draft = self.draft_store.current
        if draft is None:
            raise ValueError("manager must OPEN_STATE before COMMIT_STATE")
        traced_step_ids = tuple(draft.traced_step_ids)
        state = self.draft_store.close()
        self.state_store.commit(state)
        self.graph.add_state(state)
        self.trace_buffer.accept(traced_step_ids)
        return state

    def _relation_hint(self, hint_state_ids: tuple[str, ...]) -> str:
        if not hint_state_ids:
            return ""
        states = [self.state_store.get(item).as_state_hint() for item in hint_state_ids]
        return (
            "<analytical_state_hint>\n"
            + json.dumps(states, ensure_ascii=False, indent=2, default=str)
            + "\n</analytical_state_hint>\n"
            "These manager-selected states are observations. For reference only. Verify them before use."
        )

    @staticmethod
    def _relation_ids(relations) -> list[str]:
        return [
            relation.related_state_id
            for relation in relations
            if relation.related_state_id is not None
        ]
