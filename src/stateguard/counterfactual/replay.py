from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from threading import Lock
from typing import Any

from stateguard.agents.react import parse_action, parse_json_object
from stateguard.core.models import Message
from stateguard.providers.base import ModelResponse
from stateguard.validation.models import ManagerAction, ManagerDecision

from .hashing import messages_digest
from .models import InterventionPlan, ReplayManifest


_WORKER_DATA_ROOT = re.compile(r"(?:/[A-Za-z0-9_.:@+~-]+)+/worker_data(?=/|\b)")


class ReplayMismatchError(RuntimeError):
    """The current clean prefix no longer matches the recorded parent run."""


class ForcedManagerSequenceError(RuntimeError):
    """A curated Manager response was rejected or consumed out of order."""


@dataclass
class ReplaySwitch:
    """Shared branch-local switch activated only by the targeted Worker call."""

    activated: bool = False
    intervention_id: str | None = None
    pending_repair_requests: int = 0
    applied_repair_requests: int = 0
    forced_manager_repair_pending: bool = False
    forced_manager_repairs_applied: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def activate(self, intervention_id: str) -> None:
        with self._lock:
            if self.activated and self.intervention_id != intervention_id:
                raise RuntimeError("replay switch was activated by another intervention")
            self.activated = True
            self.intervention_id = intervention_id

    def on_repair_applied(self) -> None:
        with self._lock:
            if not self.activated:
                raise RuntimeError("Manager cannot repair before Worker intervention")
            self.pending_repair_requests += 1
            self.applied_repair_requests += 1
            if self.forced_manager_repair_pending:
                self.forced_manager_repairs_applied += 1
                self.forced_manager_repair_pending = False

    def mark_forced_manager_repair(self) -> None:
        with self._lock:
            self.forced_manager_repair_pending = True

    def clear_forced_manager_repair(self) -> None:
        with self._lock:
            self.forced_manager_repair_pending = False

    def consume_repair_request(self) -> bool:
        with self._lock:
            if self.pending_repair_requests < 1:
                return False
            self.pending_repair_requests -= 1
            return True


class ReplayThenLiveWorkerBackend:
    """Replay a verified Worker prefix, inject once, then use only the live backend."""

    def __init__(
        self,
        manifest: ReplayManifest,
        switch: ReplaySwitch,
        *,
        live_backend: Any | None = None,
        intervention: InterventionPlan | None = None,
    ) -> None:
        self.manifest = manifest
        self.switch = switch
        self.live_backend = live_backend
        self.intervention = intervention
        self.call_index = 0
        self.replayed_calls = 0
        self.live_calls = 0
        self.forced_repair_calls = 0
        self._parent_worker_data = str(Path(manifest.parent_run) / "worker_data")
        self.injected_call_index: int | None = None
        self._forced_repairs = list(
            intervention.forced_repair_responses if intervention is not None else ()
        )
        self._forced_repair_sequence_active = False
        self._target_index = None
        if intervention is not None:
            target = manifest.worker_call(
                intervention.target_unit_id, intervention.target_step_id
            )
            if intervention.replacement_action == target.response:
                raise ValueError("counterfactual replacement equals the clean Worker action")
            self._target_index = target.call_index

    def generate(self, conversation: list[dict[str, str]]) -> str:
        if self.switch.activated:
            repair_requested = self.switch.consume_repair_request()
            if repair_requested and self._forced_repairs:
                self._forced_repair_sequence_active = True
            if self._forced_repair_sequence_active:
                self.forced_repair_calls += 1
                response = self._rewrite_recorded_response_paths(
                    self._forced_repairs.pop(0), conversation
                )
                if not self._forced_repairs:
                    self._forced_repair_sequence_active = False
                return response
            return self._generate_live(conversation)

        next_index = self.call_index + 1
        if next_index > len(self.manifest.worker_calls):
            raise ReplayMismatchError("Worker clean replay tape was exhausted")
        expected = self.manifest.worker_calls[next_index - 1]
        actual_hash = messages_digest(conversation)
        if actual_hash != expected.input_hash:
            raise ReplayMismatchError(
                f"Worker input mismatch at call {next_index} "
                f"({expected.unit_id} step {expected.step_id})"
            )
        self.call_index = next_index
        if self._target_index == next_index:
            assert self.intervention is not None
            self.injected_call_index = next_index
            self.switch.activate(self.intervention.intervention_id)
            return self._rewrite_recorded_response_paths(
                self.intervention.replacement_action, conversation
            )
        self.replayed_calls += 1
        return self._rewrite_recorded_response_paths(expected.response, conversation)

    def _rewrite_recorded_response_paths(
        self, response: str, conversation: list[dict[str, str]]
    ) -> str:
        """Execute replayed Worker code against the branch-local staged data.

        Parent Worker responses often hard-code the run-local ``worker_data``
        path they saw during the clean run. Hashing already normalizes those
        paths, but execution must be redirected too; otherwise replayed code can
        inspect the finished parent run directory and observe artifacts that did
        not exist when the clean prefix originally ran.
        """
        current_data_root = self._current_worker_data_root(conversation)
        if not current_data_root:
            return response
        rewritten = response.replace(self._parent_worker_data, current_data_root)
        return _WORKER_DATA_ROOT.sub(current_data_root, rewritten)

    @staticmethod
    def _current_worker_data_root(conversation: list[dict[str, str]]) -> str | None:
        for message in conversation:
            if message.get("role") != "system":
                continue
            match = _WORKER_DATA_ROOT.search(str(message.get("content", "")))
            if match:
                return match.group(0)
        return None

    def _generate_live(self, conversation: list[dict[str, str]]) -> str:
        if self.live_backend is None:
            raise ReplayMismatchError(
                "Worker requested a live suffix, but no live backend was configured"
            )
        self.live_calls += 1
        return str(self.live_backend.generate(conversation))


class ReplayThenLiveManagerClient:
    """Replay a clean prefix, optionally demonstrate repair, then continue live.

    Curated responses use peek/ack/reject semantics: delivery does not remove a
    response. A successful Tool result or harness-accepted control action must
    acknowledge it before the following response can be delivered.
    """

    def __init__(
        self,
        manifest: ReplayManifest,
        switch: ReplaySwitch,
        *,
        live_client: Any | None = None,
        intervention: InterventionPlan | None = None,
    ) -> None:
        self.manifest = manifest
        self.switch = switch
        self.live_client = live_client
        self.intervention = intervention
        self.call_index = 0
        self.replayed_calls = 0
        self.live_calls = 0
        self.forced_manager_calls = 0
        self.forced_manager_responses_accepted = 0
        self.forced_manager_repair_calls = 0
        self.forced_manager_rejections = 0
        self.forced_manager_sequence_error: str | None = None
        self.forced_manager_sequence_started = False
        self.forced_manager_sequence_completed = False
        self.forced_post_repair_manager_calls = 0
        self._forced_repair_waiting_application = False
        self._post_repair_manager_active = False
        self._inflight_response: str | None = None
        self._inflight_action = None
        self._forced_responses = list(
            intervention.forced_manager_responses if intervention is not None else ()
        )
        self._forced_post_repair_responses = list(
            intervention.forced_post_repair_manager_responses
            if intervention is not None
            else ()
        )

    def complete(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        if self.switch.activated:
            self._raise_if_forced_sequence_failed()
            self._resolve_inflight_tool(messages)
            self._raise_if_forced_sequence_failed()
            if self._forced_responses:
                if self._inflight_response is not None:
                    raise ForcedManagerSequenceError(
                        "the previous forced Manager response was not acknowledged"
                    )
                response = self._forced_responses[0]
                action = parse_action(response)
                self._inflight_response = response
                self._inflight_action = action
                self.forced_manager_sequence_started = True
                self.forced_manager_calls += 1
                if self._post_repair_manager_active:
                    self.forced_post_repair_manager_calls += 1
                decision = _manager_decision(response)
                if decision is not None and decision.action is ManagerAction.REPAIR:
                    self.forced_manager_repair_calls += 1
                return ModelResponse(
                    response,
                    "",
                    {"forced_teacher": True, "intervention_id": self.switch.intervention_id},
                )
            if self.live_client is None:
                raise ReplayMismatchError(
                    "Manager requested a live suffix, but no live client was configured"
                )
            self.live_calls += 1
            return self.live_client.complete(messages, tools)

        next_index = self.call_index + 1
        if next_index > len(self.manifest.manager_calls):
            raise ReplayMismatchError("Manager clean replay tape was exhausted")
        expected = self.manifest.manager_calls[next_index - 1]
        actual_hash = messages_digest(messages)
        if actual_hash != expected.input_hash:
            raise ReplayMismatchError(
                f"Manager input mismatch at call {next_index} "
                f"({expected.event_type or 'UNKNOWN'})"
            )
        self.call_index = next_index
        self.replayed_calls += 1
        return ModelResponse(expected.response, expected.reasoning, {"replayed": True})

    def on_manager_control_accepted(self, action_name: str) -> None:
        """Acknowledge one forced control after lifecycle preflight accepts it."""
        if self._inflight_response is None:
            return
        action = self._inflight_action
        if action is None or action.kind != "control":
            self._reject_forced_response(
                "harness accepted a control while the forced response was not control"
            )
            raise ForcedManagerSequenceError(self.forced_manager_sequence_error or "")
        decision = _manager_decision(self._inflight_response)
        assert decision is not None
        if decision.action.value != action_name:
            self._reject_forced_response(
                f"harness accepted {action_name}, expected {decision.action.value}"
            )
            raise ForcedManagerSequenceError(self.forced_manager_sequence_error or "")
        is_repair = decision.action is ManagerAction.REPAIR
        self._ack_forced_response()
        if is_repair:
            self._forced_repair_waiting_application = True
            self.switch.mark_forced_manager_repair()

    def on_manager_control_rejected(self, action_name: str, reason: str) -> None:
        """Invalidate a started forced sequence; never advance to its next item."""
        if not self.forced_manager_sequence_started:
            return
        if self.forced_manager_sequence_completed:
            return
        self._reject_forced_response(f"{action_name}: {reason}")

    def on_repair_applied(self) -> None:
        """Forward Worker retry notification and activate optional stabilization."""
        self.switch.on_repair_applied()
        if self._forced_repair_waiting_application:
            self._forced_repair_waiting_application = False
            if self._forced_post_repair_responses:
                self._forced_responses.extend(self._forced_post_repair_responses)
                self._forced_post_repair_responses.clear()
                self._post_repair_manager_active = True
            else:
                self.forced_manager_sequence_completed = True

    def _resolve_inflight_tool(self, messages: list[Message]) -> None:
        if self._inflight_response is None:
            return
        action = self._inflight_action
        if action is None or action.kind != "tool":
            self._reject_forced_response(
                "another Manager model call was requested before the forced control "
                "action was acknowledged"
            )
            return
        if not messages:
            self._reject_forced_response("forced Tool action produced no tool result")
            return
        try:
            result = parse_json_object(messages[-1].content)
        except Exception as exc:
            self._reject_forced_response(
                f"forced Tool action produced an unreadable result: {type(exc).__name__}: {exc}"
            )
            return
        if result.get("tool") != action.tool_name:
            self._reject_forced_response(
                f"forced Tool result mismatch: expected {action.tool_name}, "
                f"got {result.get('tool')}"
            )
            return
        if result.get("ok") is not True:
            self._reject_forced_response(
                f"forced Tool {action.tool_name} failed: {result.get('error') or 'ok=false'}"
            )
            return
        self._ack_forced_response()

    def _ack_forced_response(self) -> None:
        if self._inflight_response is None or not self._forced_responses:
            raise ForcedManagerSequenceError("no forced Manager response is awaiting ack")
        if self._forced_responses[0] != self._inflight_response:
            raise ForcedManagerSequenceError("forced Manager response queue lost ordering")
        self._forced_responses.pop(0)
        self.forced_manager_responses_accepted += 1
        self._inflight_response = None
        self._inflight_action = None
        if self._post_repair_manager_active and not self._forced_responses:
            self._post_repair_manager_active = False
            self.forced_manager_sequence_completed = True

    def _reject_forced_response(self, reason: str) -> None:
        if self.forced_manager_sequence_error is not None:
            return
        self.forced_manager_rejections += 1
        self.forced_manager_sequence_error = reason
        self._forced_repair_waiting_application = False
        self.switch.clear_forced_manager_repair()

    def _raise_if_forced_sequence_failed(self) -> None:
        if self.forced_manager_sequence_error is not None:
            raise ForcedManagerSequenceError(self.forced_manager_sequence_error)


def replay_report(
    worker: ReplayThenLiveWorkerBackend,
    manager: ReplayThenLiveManagerClient,
) -> dict[str, Any]:
    return {
        "switch_activated": worker.switch.activated,
        "intervention_id": worker.switch.intervention_id,
        "worker": {
            "recorded_calls_consumed": worker.call_index,
            "replayed_calls": worker.replayed_calls,
            "injected_call_index": worker.injected_call_index,
            "forced_repair_calls": worker.forced_repair_calls,
            "forced_repairs_remaining": len(worker._forced_repairs),
            "pending_repair_requests": worker.switch.pending_repair_requests,
            "live_calls": worker.live_calls,
        },
        "manager": {
            "recorded_calls_consumed": manager.call_index,
            "replayed_calls": manager.replayed_calls,
            "forced_manager_calls": manager.forced_manager_calls,
            "forced_manager_responses_accepted": (
                manager.forced_manager_responses_accepted
            ),
            "forced_manager_repair_calls": manager.forced_manager_repair_calls,
            "forced_post_repair_manager_calls": (
                manager.forced_post_repair_manager_calls
            ),
            "forced_manager_responses_remaining": len(manager._forced_responses),
            "forced_post_repair_manager_responses_remaining": len(
                manager._forced_post_repair_responses
            ),
            "forced_manager_rejections": manager.forced_manager_rejections,
            "forced_manager_sequence_error": manager.forced_manager_sequence_error,
            "forced_manager_sequence_completed": (
                manager.forced_manager_sequence_completed
            ),
            "forced_manager_repairs_applied": (
                manager.switch.forced_manager_repairs_applied
            ),
            "live_calls": manager.live_calls,
        },
    }


def _manager_decision(response: str) -> ManagerDecision | None:
    """Parse a Manager control response; Tool actions intentionally return None."""
    action = parse_action(response)
    if action.kind != "control":
        return None
    return ManagerDecision.from_dict(parse_json_object(action.answer or ""))
