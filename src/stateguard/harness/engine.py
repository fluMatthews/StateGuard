from __future__ import annotations

import json
from urllib.error import HTTPError
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
from stateguard.state.draft import DraftStore, SourceInterval, StateHeader, StateUpdate
from stateguard.state.graph import StateRelationGraph
from stateguard.state.models import AnalyticalState
from stateguard.state.store import StateStore
from stateguard.telemetry.artifacts import RunArtifactWriter
from stateguard.validation.models import ManagerAction, ManagerDecision

from .blind_view import BlindViewBuilder, ManagerObservation, assert_blind


class Manager(Protocol):
    def start_task(self, task: TaskSpec) -> None: ...

    def act(self, observation: ManagerObservation) -> ManagerDecision: ...


# The server shares one 40,960-token window between the request and the reply, so
# an input that fits is not automatically an input that leaves room to answer.
# Measured with the model's own tokenizer at 3.65 characters per token: fourteen
# recorded failures all came back finish_reason="length" with total_tokens exactly
# at the window, three of them with 47 to 239 tokens left to write a whole action
# into. Capping the input at 32,000 leaves about 9,000 for the reply, and touches
# 6.2% of LongDS activations against 61.1% of DE ones, which is where every
# recorded truncation happened.
MANAGER_INPUT_TOKEN_CAP = 32_000
MANAGER_CHARS_PER_TOKEN = 3.65


def _activation_action_counts(raw_activation: dict[str, Any] | None) -> dict[str, int]:
    """Count the actions this activation completed, by kind.

    A control action is already legible in the recorded command, but a tool
    action only ever appeared inside the activation's own messages, so counting
    how many the Manager issued meant parsing JSON out of a message body. These
    are the denominators a per-kind success rate needs; the failures that never
    produced an action carry their kind on the failure record instead.
    """
    counts = {"tool": 0, "control": 0, "tool_ok": 0}
    if not isinstance(raw_activation, dict):
        return counts
    for message in raw_activation.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if role == "assistant":
            try:
                payload = json.loads(content)
            except Exception:  # noqa: BLE001 - an unparsed reply is not an action
                continue
            kind = payload.get("type")
            if kind in counts:
                counts[kind] += 1
        elif role == "user" and content.lstrip().startswith("<tool_result>"):
            try:
                body = json.loads(content.split("<tool_result>")[1].split("</tool_result>")[0])
            except Exception:  # noqa: BLE001
                continue
            if body.get("ok"):
                counts["tool_ok"] += 1
    return counts


def _attempted_action_kind(exc: Exception) -> str | None:
    """Which kind of action the Manager was attempting when this failed.

    parse_action reads the JSON before it reads the type, so a malformed
    envelope raises with the kind still unknown and every format failure looked
    alike. The unparsed text is carried on the error, and the opening
    {"type":"..." sits ahead of the reasoning field where a truncation lands, so
    the intent survives even when the object does not.
    """
    raw = getattr(exc, "raw_response", None)
    if not isinstance(raw, str):
        return None
    head = "".join(raw[:200].split())
    if '"type":"tool"' in head or "'type':'tool'" in head:
        return "tool"
    if '"type":"control"' in head or "'type':'control'" in head:
        return "control"
    return None


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
        self.manager_probe_executor = self.runtime.manager_probe_executor
        self.checkpoints = self.runtime.checkpoints
        self.blind_view = BlindViewBuilder()
        self._manager_failures: list[ManagerFailure] = []
        self._manager_action_index: int | None = None
        self._worker_steps_this_run = 0
        self._terminal_state_id: str | None = None
        self._terminal_pending_step_ids: tuple[int, ...] = ()
        self._deferred_action_result: dict[str, Any] | None = None
        self._last_summary_size: int | None = None
        committed_numbers = [int(state.id[1:]) for state in self.state_store.all()]
        self._next_state_number = max(committed_numbers, default=0) + 1

    def run(self, task: TaskSpec) -> StateGuardResult:
        self._manager_failures = []
        self._manager_action_index = None
        self._worker_steps_this_run = 0
        self._terminal_state_id = None
        self._terminal_pending_step_ids = ()
        self._deferred_action_result = None
        self._last_summary_size = None
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
            if self._automates_turn_start():
                self._record_mandatory_preopen_failure(task, exc)
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
        mandatory_preopen_failed = False
        manager_cycle_due = self.flow_adapter.review_before_worker()

        while True:
            if manager_cycle_due:
                try:
                    outcome, action_count, repair_delta, abstain_delta = self._manager_cycle(
                        task=task,
                        event_type=event_type,
                        repair_session=repair_session,
                    )
                except Exception as exc:
                    self._notify_manager_control_rejected(
                        "UNKNOWN", f"act_or_execute: {type(exc).__name__}: {exc}"
                    )
                    self._record_manager_failure(
                        phase="act_or_execute",
                        event_type=event_type,
                        exc=exc,
                        action_index=self._manager_action_index,
                    )
                    total_manager_actions += 1
                    if self.worker.done:
                        break
                    if event_type == "TASK_START" and self._automates_turn_start():
                        self._record_mandatory_preopen_failure(task, exc)
                        mandatory_preopen_failed = True
                        outcome, action_count, repair_delta, abstain_delta = (
                            "RUN_TO_COMPLETION",
                            0,
                            0,
                            0,
                        )
                    else:
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
            if outcome == "RUN_TO_COMPLETION":
                self.artifacts.record(
                    "trace",
                    {
                        "event": (
                            "mandatory_preopen_failed_worker_handoff"
                            if mandatory_preopen_failed
                            else "terminal_state_manager_handoff"
                        )
                    },
                )
                while not self.worker.done:
                    if total_worker_steps >= self.config.max_worker_steps:
                        raise RuntimeError(
                            f"StateGuard exceeded max_worker_steps={self.config.max_worker_steps}"
                        )
                    latest_step = self.worker.step()
                    total_worker_steps += 1
                    self._worker_steps_this_run = total_worker_steps
                    self.trace_buffer.append(
                        latest_step, repair_attempt=repair_session.attempts
                    )
                    self.artifacts.record("worker", latest_step)
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
                self._worker_steps_this_run = total_worker_steps
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
        repair_session: RepairSession,
    ) -> tuple[str, int, int, int]:
        if self.manager is None:
            raise RuntimeError("manager cycle is unavailable when manager=None")
        last_action_result = self._deferred_action_result
        self._deferred_action_result = None
        repairs = 0
        abstained = 0
        preflight_retry_used = False

        for action_index in range(1, self.config.max_manager_actions_per_event + 1):
            self._mark_terminal_pending_review()
            self._manager_action_index = action_index
            observation = self._manager_observation(
                task, event_type, repair_session, last_action_result
            )
            decision = self._act_within_context_window(
                task, event_type, repair_session, last_action_result, observation
            )
            raw_activation = self._manager_activation_snapshot()
            command = self._canonicalize_command(decision)
            self.artifacts.record(
                "manager",
                {
                    "event_type": event_type,
                    "observation": observation,
                    "command": command,
                    "raw_activation": raw_activation,
                    "action_counts": _activation_action_counts(raw_activation),
                },
            )
            try:
                if (
                    event_type == "TASK_START"
                    and self._automates_turn_start()
                    and command.action is not ManagerAction.OPEN_STATE
                ):
                    raise ValueError("multi-turn TASK_START requires OPEN_STATE")
                self._preflight_action(command, repair_session)
            except (KeyError, TypeError, ValueError) as exc:
                self._notify_manager_control_rejected(
                    command.action.value,
                    f"preflight: {type(exc).__name__}: {exc}",
                )
                mandatory_preopen = (
                    event_type == "TASK_START" and self._automates_turn_start()
                )
                if preflight_retry_used:
                    raise
                last_action_result = {
                    "action": "ACTION_REJECTED",
                    "requested_action": command.action.value,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retry_allowed": not mandatory_preopen,
                }
                guidance = self._rejection_next_step(command.action)
                if guidance:
                    last_action_result["next_step"] = guidance
                self.artifacts.record(
                    "manager",
                    {
                        "event_type": event_type,
                        "event": "action_preflight_rejected",
                        **last_action_result,
                    },
                )
                if mandatory_preopen:
                    raise
                preflight_retry_used = True
                event_type = "ACTION_RESULT"
                continue
            self._notify_manager_control_accepted(command.action.value)
            preflight_retry_used = False

            if command.action is ManagerAction.RESUME_WORKER:
                self._resume_state_summary()
                return "RESUME", action_index, repairs, abstained

            if command.action is ManagerAction.OPEN_STATE:
                opens_mandatory_turn = (
                    event_type == "TASK_START" and self._automates_turn_start()
                )
                header = command.state_header
                assert header is not None
                opens_terminal_state = bool(self._terminal_pending_step_ids)
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
                if opens_terminal_state:
                    self._terminal_state_id = header.id
                    pending_step_ids = self._terminal_pending_step_ids
                    self._terminal_pending_step_ids = ()
                    self.artifacts.record(
                        "state",
                        {
                            "event": "terminal_state_started",
                            "state_id": header.id,
                            "pending_step_ids": list(pending_step_ids),
                        },
                    )
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
                if opens_mandatory_turn:
                    # OPEN_STATE is the only semantic turn-start decision. Keep
                    # its result for turn-end review and resume mechanically.
                    self._deferred_action_result = last_action_result
                    return "RESUME", action_index, repairs, abstained
                event_type = "ACTION_RESULT"
                continue

            if command.action is ManagerAction.UPDATE_STATE:
                with self._action_transaction(
                    "update_state",
                    components=("state_draft", "trace_buffer"),
                ):
                    draft = self._update_state(
                        command.state_update,
                        allow_sparse_trace=self._manager_selects_source_interval(),
                    )
                last_action_result = {
                    "action": "UPDATE_STATE",
                    "state_id": draft.header.id,
                    "source_interval": (
                        draft.source_interval.to_dict()
                        if draft.source_interval is not None
                        else None
                    ),
                }
                self._attach_constraint_checks(draft, last_action_result)
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
                            allow_sparse_trace=self._manager_selects_source_interval(),
                        )
                    state = self._commit_draft()
                    checkpoint = self.checkpoints.capture(f"state:{state.id}")
                    repair_session.complete_state(checkpoint)
                commits_terminal_state = state.id == self._terminal_state_id
                if commits_terminal_state:
                    terminal_tail = self.trace_buffer.pass_uncommitted()
                    if terminal_tail:
                        self.artifacts.record(
                            "trace",
                            {
                                "event": "terminal_unselected_steps_passed",
                                "step_ids": list(terminal_tail),
                            },
                        )
                    self._terminal_state_id = None
                self._release_checkpoint_refs(*retired_checkpoints)
                self.artifacts.record(
                    "trace",
                    {"event": "state_committed", "history": self.trace_buffer.history()},
                )
                last_action_result = {"action": "COMMIT_STATE", "state_id": state.id}
                if commits_terminal_state:
                    outcome = "FINISH" if self.worker.done else "RUN_TO_COMPLETION"
                    return outcome, action_index, repairs, abstained
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
                        if self._reviews_terminal_pending():
                            rejected_step_ids = self.trace_buffer.reject_uncommitted()
                        else:
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
                intervention_observer = getattr(
                    self.manager, "intervention_observer", None
                )
                on_repair_applied = getattr(
                    intervention_observer, "on_repair_applied", None
                )
                if callable(on_repair_applied):
                    on_repair_applied()
                repairs += 1
                return "RESUME", action_index, repairs, abstained

            if command.action is ManagerAction.ABANDON_STATE:
                state_id = repair_session.state_id
                self.repair_controller.abandon_state(repair_session, self.checkpoints)
                repair_record = repair_session.records[-1]
                passed_step_ids = self._settle_abandoned_state(repair_session)
                if state_id == self._terminal_state_id:
                    self._terminal_state_id = None
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
                terminal_state_id = self._terminal_state_id
                self._terminal_pending_step_ids = ()
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
                if terminal_state_id is not None:
                    self._terminal_state_id = None
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

    def _manager_activation_snapshot(self) -> dict[str, Any] | None:
        export_session = getattr(self.manager, "export_session", None)
        if not callable(export_session):
            return None
        session = export_session()
        messages = session.get("messages") if isinstance(session, dict) else None
        if not isinstance(messages, list):
            return None
        start = 0
        for index in range(len(messages) - 1, -1, -1):
            metadata = messages[index].get("metadata", {})
            if isinstance(metadata, dict) and metadata.get("manager_block_start"):
                start = index
                break
        return {
            "messages": messages[start:],
            "step_count": session.get("step_count"),
            "done": session.get("done"),
            "final_answer": session.get("final_answer"),
        }

    def _notify_manager_control_accepted(self, action_name: str) -> None:
        observer = getattr(self.manager, "intervention_observer", None)
        callback = getattr(observer, "on_manager_control_accepted", None)
        if callable(callback):
            callback(action_name)

    def _notify_manager_control_rejected(
        self, action_name: str, reason: str
    ) -> None:
        observer = getattr(self.manager, "intervention_observer", None)
        callback = getattr(observer, "on_manager_control_rejected", None)
        if callable(callback):
            callback(action_name, reason)

    def _attach_constraint_checks(
        self, draft: Any, result: dict[str, Any]
    ) -> None:
        """Run the checking code a constraint carries, once the body exists.

        A constraint may carry code, and nothing ever ran it, so "verified"
        stayed a word the Manager wrote rather than a result it obtained: one
        state asserted row counts consistent with the contract while every table
        sat one row short of its source. This runs at the point the lifecycle
        already reserves for checking the written body and choosing between
        REPAIR and FINALIZE_RELATIONS, so the outcome arrives where the decision
        is made. A check that fails to run at all is not recorded: the code may
        address something absent, and an unverified constraint reported as a
        violation would be worse than one left unchecked.
        """
        executor = self.manager_probe_executor
        if executor is None:
            return
        checks: list[dict[str, str]] = []
        failed: list[dict[str, str]] = []
        for constraint in draft.header.constraints:
            code = getattr(constraint, "code", None)
            if not code or not str(code).strip():
                continue
            try:
                outcome = executor.execute(str(code))
            except Exception:  # noqa: BLE001 - a check must never end the review
                continue
            if outcome.ok:
                checks.append({"text": constraint.text, "status": "pass"})
                continue
            detail = str(outcome.error or outcome.stdout or "")
            if "AssertionError" not in detail:
                # The code never reached its assertion, so it settles nothing.
                continue
            entry = {
                "text": constraint.text,
                "status": "fail",
                "output": detail.strip().splitlines()[-1][:400],
            }
            checks.append(entry)
            failed.append(entry)
        if not checks:
            return
        result["constraint_checks"] = checks
        if failed:
            first = failed[0]
            result["next_step"] = (
                f'constraint "{first["text"]}" did not hold: {first["output"]}. '
                "REPAIR if the evidence localizes the error."
            )

    def _rejection_next_step(self, action: ManagerAction) -> str | None:
        """Name the action the rejected one should have been, when the rejection
        was a stage mismatch.

        A rejection used to reach the Manager as its error string alone, so a
        Manager that opened a state it had already opened learned only that the
        state was open, not which of write, finalize or commit it still owed.
        The draft carries that: an unwritten body has no source_interval, and an
        unselected relation set is not finalized. A rejection about the content
        of an otherwise well-placed action gets nothing from here -- its own
        error already says what is wrong, and a stage sentence would mislead.
        Only the two payload-free actions are shown in full, because the flow
        lifecycle, not this harness, defines what an OPEN_STATE, UPDATE_STATE or
        FINALIZE_RELATIONS body may carry.
        """
        draft = self.draft_store.current
        if draft is None:
            if action is ManagerAction.REPAIR:
                return (
                    "No state is open, so there is nothing to repair. Resume the "
                    "Worker and carry the doubt into the state you open next: "
                    '{"type":"control","reasoning":"...","answer":'
                    '{"action":"RESUME_WORKER"}}'
                )
            if action in {
                ManagerAction.UPDATE_STATE,
                ManagerAction.FINALIZE_RELATIONS,
                ManagerAction.COMMIT_STATE,
                ManagerAction.ABANDON_STATE,
            }:
                return (
                    "No state is open, so there is no body to act on. OPEN_STATE "
                    "first, using the header shape in section 5.2."
                )
            return None
        if action is ManagerAction.OPEN_STATE:
            return self._open_draft_next_step(draft)
        if action is ManagerAction.COMMIT_STATE and not draft.relations_finalized:
            return self._open_draft_next_step(draft)
        return None

    @staticmethod
    def _open_draft_next_step(draft: Any) -> str:
        """What the open draft still owes, without foreclosing a repair.

        Once the body is written the choice is the one section 6 describes --
        REPAIR when the checks evidenced a violation, otherwise carry on -- so
        naming only the forward action here would talk the Manager out of a
        repair it was entitled to make.
        """
        if draft.source_interval is None:
            return (
                f"{draft.header.id} is open and its body is not written yet. "
                "Write it with UPDATE_STATE, using the shape in section 5.3."
            )
        if not draft.relations_finalized:
            return (
                f"{draft.header.id} is open and its body is written, so check it "
                "and take one of the two exits: REPAIR if the checks evidenced a "
                "violation, otherwise FINALIZE_RELATIONS in the mode and shape "
                "section 5.4 allows."
            )
        return (
            f"{draft.header.id} is open, written and its relations are selected. "
            "Commit it unless the checks still evidence a violation, which REPAIR "
            'answers instead: {"type":"control","reasoning":"...","answer":'
            '{"action":"COMMIT_STATE"}}'
        )

    @staticmethod
    def _is_context_overflow(exc: Exception) -> bool:
        """Whether the server refused the request for length rather than content.

        Only a 400 that names length qualifies. A 400 whose body says anything
        else, or says nothing at all, is a real rejection: truncating the trace
        would discard Worker evidence to retry a request that was never too long.
        The body is read once and cached on the exception, since ``HTTPError.read``
        drains it and a caller may have already looked.
        """
        if not isinstance(exc, HTTPError) or exc.code != 400:
            return False
        detail = getattr(exc, "_stateguard_detail", None)
        if detail is None:
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 - a body is a courtesy, not a guarantee
                detail = ""
            try:
                exc._stateguard_detail = detail
            except Exception:  # noqa: BLE001 - exotic exception types may reject it
                pass
        haystack = f"{detail} {exc.reason}".lower()
        return (
            "maximum context length" in haystack
            or "context length" in haystack
            or "reduce the length" in haystack
            or "too long" in haystack
        )

    def _act_within_context_window(
        self,
        task: TaskSpec,
        event_type: str,
        repair_session: RepairSession,
        last_action_result: dict[str, Any] | None,
        observation: ManagerObservation,
    ):
        """Retry a refused-for-length activation on a halved pending trace.

        A single-query unit never resets its pending trace, so an overflow is not
        transient: the failure leaves pending intact, the Worker adds more steps,
        and every later pause overflows again. Halving keeps the newest steps --
        the ones the next state would cover -- and repeats until the request fits
        or nothing is left, so a unit that overflowed once can still be reviewed.
        """
        while self._observation_exceeds_input_cap(observation):
            if not self._shrink_pending_once(event_type):
                break
            observation = self._manager_observation(
                task, event_type, repair_session, last_action_result
            )
        while True:
            try:
                return self.manager.act(observation)
            except Exception as exc:  # noqa: BLE001 - re-raised unless it fits
                if not self._is_context_overflow(exc):
                    raise
                if not self._shrink_pending_once(event_type):
                    raise
                observation = self._manager_observation(
                    task, event_type, repair_session, last_action_result
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
            action_kind=_attempted_action_kind(exc),
        )
        self._manager_failures.append(failure)
        self.artifacts.record("manager_failure", failure)
        # A parse failure is raised before the response reaches the session, so
        # ``export_session`` below carries the previous successful action rather
        # than the one that failed. Record the unparsed text separately.
        raw_response = getattr(exc, "raw_response", None)
        if raw_response is not None:
            self.artifacts.record(
                "manager_parse_failure",
                {"failure": failure, "raw_response": raw_response},
            )
        export_session = getattr(self.manager, "export_session", None)
        if callable(export_session):
            self.artifacts.record(
                "manager_raw",
                {
                    "failure": failure,
                    "session": export_session(),
                },
            )

    def _observation_exceeds_input_cap(
        self, observation: ManagerObservation
    ) -> bool:
        """Whether this observation would crowd the reply out of the window.

        select_manager_context already budgets the session, but it can only drop
        historical blocks: the pinned prompt and the newest block -- this
        observation -- go in at whatever size they are. An observation large
        enough on its own therefore passed the server, which refuses only an
        illegal request, and then left too few tokens for the answer. The reply
        stopped inside its own reasoning field and came back as a parse error
        that read like a formatting fault rather than a length one.
        """
        pinned = sum(
            len(message.content) + len(message.name or "")
            for message in tuple(getattr(self.manager, "messages", ()))[:2]
        )
        rendered = len(
            json.dumps(observation.to_dict(), ensure_ascii=False, default=str)
        )
        budget = MANAGER_INPUT_TOKEN_CAP * MANAGER_CHARS_PER_TOKEN
        return (pinned + rendered) > budget

    def _shrink_pending_once(self, event_type: str) -> bool:
        """Shorten the pending steps, or halve them, or report nothing is left."""
        steps = self.trace_buffer.steps
        pending = len(steps)
        if pending == 0:
            return False
        if self._elide_pending_steps(steps, event_type):
            return True
        keep = (pending + 1) // 2 if pending > 1 else 0
        dropped = self.trace_buffer.drop_oldest_pending(keep)
        if not dropped:
            return False
        self.artifacts.record(
            "trace",
            {
                "event": "pending_truncated_for_context",
                "dropped_step_ids": list(dropped),
                "kept_pending": keep,
                "event_type": event_type,
            },
        )
        return True

    def _elide_pending_steps(self, steps, event_type: str) -> bool:
        """Ask the Worker to shorten every pending step; report whether any changed.

        Shortening comes before dropping because dropping is the lossier move: a
        Worker whose steps are large enough to overflow on their own gets halved
        all the way down to one step, and a review of one step forms no state.
        Trading the middle of the bulkiest fields keeps the whole window of steps
        the next state would cover. A step is only ever shortened once -- the
        Worker returns None for one it has already rewritten -- so a later
        overflow that finds nothing left to shorten falls through to halving and
        the retry loop still terminates.

        Workers that do not offer ``elide_step`` keep the drop-only behaviour.
        """
        elide = getattr(self.worker, "elide_step", None)
        if not callable(elide):
            return False
        elided: list[int] = []
        for step in steps:
            shortened = elide(step)
            if shortened is None:
                continue
            self.trace_buffer.replace_pending_step(shortened)
            elided.append(step.step_id)
        if not elided:
            return False
        self.artifacts.record(
            "trace",
            {
                "event": "pending_steps_elided_for_context",
                "step_ids": elided,
                "pending": len(steps),
                "event_type": event_type,
            },
        )
        return True

    def _manager_observation(
        self,
        task: TaskSpec,
        event_type: str,
        repair_session: RepairSession,
        last_action_result: dict[str, Any] | None,
    ) -> ManagerObservation:
        draft = self.draft_store.current
        pending_steps = tuple(self.trace_buffer.steps)
        terminal_flow = self._reviews_terminal_pending()
        include_state_index = bool(
            (event_type == "TASK_START" and self._is_turn_mode())
            or (
                last_action_result is not None
                and last_action_result.get("action") == "UPDATE_STATE"
            )
        )
        committed_state_index = (
            tuple(self.state_store.load_state_index_json())
            if include_state_index
            else None
        )
        return self.blind_view.build(
            task=task,
            event_type=event_type,
            available_state_id=f"S{self._next_state_number}",
            untraced_steps=pending_steps,
            current_draft=draft.to_observation_dict() if draft else None,
            repair_attempts=repair_session.attempts,
            terminal_pending_review=bool(
                terminal_flow
                and (self._terminal_pending_step_ids or self._terminal_state_id)
            ),
            worker_steps_remaining=(
                self._worker_steps_remaining() if terminal_flow else None
            ),
            last_action_result=last_action_result,
            committed_state_index=committed_state_index,
        )

    def _canonicalize_command(self, command: ManagerDecision) -> ManagerDecision:
        """Fill only fields owned unambiguously by the harness."""
        if command.action is ManagerAction.OPEN_STATE:
            header = command.state_header
            if header is not None and header.id == "$AUTO_STATE":
                command = replace(
                    command,
                    state_header=replace(
                        header, id=f"S{self._next_state_number}"
                    ),
                )

        if command.action is ManagerAction.FINALIZE_RELATIONS:
            finalization = command.relation_finalization
            draft = self.draft_store.current
            if (
                finalization is not None
                and finalization.mode.value == "confirm"
                and not finalization.relations
                and draft is not None
            ):
                command = replace(
                    command,
                    relation_finalization=replace(
                        finalization, relations=draft.header.relations
                    ),
                )
        if command.action in {ManagerAction.UPDATE_STATE, ManagerAction.COMMIT_STATE}:
            if command.state_update is not None:
                command = replace(
                    command, state_update=self._bind_state_source(command.state_update)
                )
        return command

    def _is_turn_mode(self) -> bool:
        mode = str(self.flow_adapter.manager_context().get("mode", "")).lower()
        return mode == "turn"

    def _automates_turn_start(self) -> bool:
        hook = getattr(self.flow_adapter, "automate_turn_start", None)
        return bool(hook()) if callable(hook) else False

    def _record_mandatory_preopen_failure(
        self, task: TaskSpec, exc: Exception
    ) -> str:
        skipped_state_id = f"S{self._next_state_number}"
        self._next_state_number += 1
        self.artifacts.record(
            "state",
            {
                "event": "mandatory_preopen_failed",
                "state_id": skipped_state_id,
                "task_id": task.id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return skipped_state_id

    def _manager_selects_source_interval(self) -> bool:
        mode = str(self.flow_adapter.manager_context().get("mode", "")).lower()
        return mode in {"single_query", "fixed_step"}

    def _bind_state_source(self, update: StateUpdate) -> StateUpdate:
        """Translate Manager source semantics into harness-owned trace IDs."""
        pending_ids = tuple(step.step_id for step in self.trace_buffer.steps)
        draft = self.draft_store.current
        if self._manager_selects_source_interval():
            interval = update.source_interval
            if interval is None:
                return replace(update, traced_step_ids=())
            if draft is not None and draft.source_interval is not None:
                selected = tuple(step_id for step_id in pending_ids if step_id <= interval.end)
            else:
                selected = tuple(
                    step_id for step_id in pending_ids
                    if interval.start <= step_id <= interval.end
                )
            return replace(update, traced_step_ids=selected)

        if not pending_ids:
            return replace(update, traced_step_ids=())
        history_ids = tuple(
            record["step_id"] for record in self.trace_buffer.history()
        )
        start = (
            draft.source_interval.start
            if draft is not None and draft.source_interval is not None
            else min(history_ids)
        )
        return replace(
            update,
            source_interval=SourceInterval(start=start, end=pending_ids[-1]),
            traced_step_ids=pending_ids,
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
            self._preflight_update(command.state_update, repair_session)
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
                self._preflight_update(command.state_update, repair_session)
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
            if (
                draft.header.id == self._terminal_state_id
                and self._worker_steps_remaining() <= 0
            ):
                raise ValueError(
                    "terminal Worker budget is exhausted; the state cannot be repaired, "
                    "so ABSTAIN to discard this unresolved terminal draft and finish"
                )
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
            if (
                self._terminal_state_id is not None
                and self._worker_steps_remaining() > 0
            ):
                raise ValueError(
                    "an open terminal state must COMMIT, REPAIR, or exhaust its repair "
                    "schedule before it can be discarded"
                )
            return

        raise ValueError(f"unsupported manager action: {command.action}")

    def _preflight_update(
        self,
        update: StateUpdate | None,
        repair_session: RepairSession,
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

        pending_ids = tuple(step.step_id for step in self.trace_buffer.steps)
        interval = update.source_interval
        if self._manager_selects_source_interval():
            if interval is None:
                raise ValueError(
                    "single-query UPDATE_STATE requires source_interval with start and end"
                )
            if draft.source_interval is None:
                history = self.trace_buffer.history()
                rejected_ids = {
                    record["step_id"]
                    for record in history
                    if record["status"] == "rejected"
                }
                start_is_valid = interval.start in pending_ids or (
                    repair_session.attempts > 0
                    and repair_session.state_id == draft.header.id
                    and interval.start in rejected_ids
                )
                if not start_is_valid or interval.end not in pending_ids:
                    raise ValueError(
                        "source_interval must start at an observed live/rejected step "
                        "and end at an observed pending Worker step"
                    )
            else:
                if interval.start != draft.source_interval.start:
                    raise ValueError(
                        "repair must preserve the current state's source interval start"
                    )
                if interval.end < draft.source_interval.end:
                    raise ValueError("repair may only extend the source interval end")
                if interval.end not in pending_ids:
                    raise ValueError(
                        "an extended source_interval must end at an observed retry step"
                    )
            if not update.traced_step_ids:
                raise ValueError("source_interval contains no pending Worker steps")
            self.trace_buffer.validate_selection(
                update.traced_step_ids, allow_sparse=True
            )
        else:
            if not pending_ids:
                raise ValueError("multi-turn UPDATE_STATE requires a completed Worker turn")
            if update.traced_step_ids != pending_ids:
                raise ValueError(
                    "multi-turn state source is bound automatically to the complete turn"
                )
            self.trace_buffer.validate_selection(
                update.traced_step_ids, allow_sparse=False
            )

        # Reuse state-local validation without mutating the live draft.
        draft.apply(update)

    def _reviews_terminal_pending(self) -> bool:
        return bool(getattr(self.flow_adapter, "review_terminal_pending", False))

    def _worker_steps_remaining(self) -> int:
        remaining = getattr(self.worker, "remaining_steps", None)
        if isinstance(remaining, int):
            return max(0, remaining)
        return max(0, self.config.max_worker_steps - self._worker_steps_this_run)

    def _mark_terminal_pending_review(self) -> None:
        if not (
            self._reviews_terminal_pending()
            and self.worker.done
            and self._terminal_state_id is None
            and not self._terminal_pending_step_ids
            and self.trace_buffer.steps
        ):
            return
        draft = self.draft_store.current
        if draft is not None:
            # A draft still open when the Worker finishes IS the terminal state:
            # the Manager cannot OPEN another one, so waiting for an OPEN_STATE
            # that can never come left the tail unreviewed and silently passed.
            # Adopting the draft arms the terminal guards instead, which is the
            # point of the review -- COMMIT, REPAIR while budget remains, or
            # ABSTAIN once it is gone.
            self._terminal_state_id = draft.header.id
            self.artifacts.record(
                "state",
                {
                    "event": "terminal_state_started",
                    "state_id": draft.header.id,
                    "adopted_open_draft": True,
                    "pending_step_ids": [
                        step.step_id for step in self.trace_buffer.steps
                    ],
                },
            )
            return
        self._terminal_pending_step_ids = tuple(
            step.step_id for step in self.trace_buffer.steps
        )

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

    def _resume_state_summary(self) -> None:
        """Hand the Worker the newest committed states when a flow asks for it.

        A single-query flow decides for itself when pending steps form a state,
        so the Worker can run a long stretch receiving nothing. This sends the
        newest committed states on resume, and only after the store grew, so a
        run of resumes between two commits injects once rather than every time.
        The Manager never sees this: an injected observation is not a Worker
        step and never enters the trace buffer it reads.
        """
        hook = getattr(self.flow_adapter, "resumes_with_state_summary", None)
        if not callable(hook) or not hook():
            return
        committed = self.state_store.all()
        if not committed or len(committed) == self._last_summary_size:
            return
        recent = tuple(state.id for state in committed[-3:])
        summary = self._relation_hint(recent)
        if not summary:
            return
        self.worker.inject_observation(
            summary, metadata={"stateguard": "state_summary"}
        )
        self.artifacts.record(
            "hint",
            {
                "kind": "state_summary",
                "state_id": None,
                "related_state_ids": list(recent),
                "content": summary,
            },
        )
        self._last_summary_size = len(committed)

    def _relation_hint(self, hint_state_ids: tuple[str, ...]) -> str:
        if not hint_state_ids:
            return ""
        # A flow whose Worker rediscovers facts expensively asks for the
        # variables as well; every other flow keeps the hint as it was.
        wants_variables = getattr(
            self.flow_adapter, "state_hint_includes_variables", None
        )
        include = bool(wants_variables()) if callable(wants_variables) else False
        states = [
            self.state_store.get(item).as_state_hint(include_variables=include)
            for item in hint_state_ids
        ]
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
