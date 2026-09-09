from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from stateguard.agents.react import parse_action, parse_json_object
from stateguard.core.models import Message
from stateguard.validation.models import ManagerDecision

from .context import (
    DEFAULT_MANAGER_CONTEXT_CHARS,
    DEFAULT_MANAGER_OUTPUT_RESERVE_CHARS,
    select_manager_context,
)


def export_manager_activations(
    session: dict[str, Any],
    failures: Sequence[dict[str, Any]] = (),
    *,
    max_context_chars: int = DEFAULT_MANAGER_CONTEXT_CHARS,
    reserved_output_chars: int = DEFAULT_MANAGER_OUTPUT_RESERVE_CHARS,
    source: str = "manager_session",
    eligible_task_ids: set[str] | None = None,
) -> tuple[list[dict[str, list[dict[str, str]]]], list[dict[str, Any]]]:
    """Convert one complete Manager session into validated activation records.

    Each accepted record ends at one terminal StateGuard control action. It
    contains the same bounded block context selected by the runtime policy,
    while the complete input session remains untouched for audit.
    """
    raw_messages = session.get("messages")
    if not isinstance(raw_messages, list):
        raise TypeError(f"missing messages list: {source}")
    messages = [
        _message_from_dict(row, index, source)
        for index, row in enumerate(raw_messages)
    ]
    if len(messages) < 3 or messages[0].role != "system" or messages[1].role != "user":
        raise ValueError(f"invalid Manager session prefix: {source}")

    blocks = _activation_blocks(messages[2:])
    validations = [_validate_activation(block) for block in blocks]
    task_ids = [_activation_task_id(block) for block in blocks]
    coordinates = _activation_coordinates(blocks)
    failure_assignments, unmatched_failures = _assign_failures(
        coordinates, validations, task_ids, list(failures)
    )

    records: list[dict[str, list[dict[str, str]]]] = []
    report: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        reasons = list(validations[index])
        task_id = task_ids[index]
        if eligible_task_ids is not None and task_id not in eligible_task_ids:
            reasons.append("ineligible_task_unit")
        if index in failure_assignments:
            failure = failure_assignments[index]
            reasons.append(
                f"manager_failure:{failure.get('error_type', 'unknown')}:"
                f"{failure.get('message', '')}"
            )
        accepted = not reasons
        record_index = len(records) + 1 if accepted else None
        event_type, action_index = coordinates[index]
        report.append(
            {
                "activation_index": index + 1,
                "record_index": record_index,
                "task_id": task_id,
                "event_type": event_type,
                "action_index": action_index,
                "accepted": accepted,
                "reasons": reasons,
            }
        )
        if not accepted:
            continue

        selected = select_manager_context(
            messages[:2]
            + [message for prior in blocks[: index + 1] for message in prior],
            max_context_chars,
            reserved_output_chars,
        )
        records.append(
            {
                "messages": [
                    {
                        "role": message.role,
                        "content": _content_for_export(message),
                    }
                    for message in selected
                ]
            }
        )

    for failure in unmatched_failures:
        report.append(
            {
                "activation_index": None,
                "event_type": failure.get("event_type"),
                "action_index": failure.get("action_index"),
                "accepted": False,
                "reasons": [
                    (
                        "unmatched_manager_failure:"
                        f"{failure.get('error_type', 'unknown')}:"
                        f"{failure.get('message', '')}"
                    )
                ],
            }
        )
    return records, report


def _message_from_dict(value: Any, index: int, source: str) -> Message:
    if not isinstance(value, dict):
        raise TypeError(f"message {index} is not an object: {source}")
    role = value.get("role")
    content = value.get("content")
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"unsupported role {role!r}: {source}")
    if not isinstance(content, str) or not content:
        raise ValueError(f"empty content at message {index}: {source}")
    metadata = value.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise TypeError(f"invalid metadata at message {index}: {source}")
    return Message(role, content, value.get("name"), metadata)


def _activation_blocks(messages: list[Message]) -> list[list[Message]]:
    blocks: list[list[Message]] = []
    current: list[Message] = []
    for message in messages:
        if message.metadata.get("manager_block_start"):
            if current:
                blocks.append(current)
            current = [message]
        elif current:
            current.append(message)
    if current:
        blocks.append(current)
    return blocks


def _activation_task_id(block: list[Message]) -> str:
    if not block:
        return ""
    task_id = block[0].metadata.get("task_id")
    if task_id:
        return str(task_id)
    content = block[0].content.strip()
    if not content.startswith("<manager_observation>"):
        return ""
    try:
        payload_text = content.split("\n", 1)[1].rsplit(
            "\n</manager_observation>", 1
        )[0]
        payload = json.loads(payload_text)
    except (IndexError, TypeError, ValueError, json.JSONDecodeError):
        return ""
    return str(payload.get("task_id", "")) if isinstance(payload, dict) else ""


def _validate_activation(block: list[Message]) -> list[str]:
    reasons: list[str] = []
    if not block or not block[0].content.lstrip().startswith(
        "<manager_observation>"
    ):
        return ["block_does_not_start_with_manager_observation"]
    if any(
        message.role == "user"
        and message.content.lstrip().startswith("<manager_protocol_error>")
        for message in block
    ):
        reasons.append("manager_protocol_error")

    parsed_actions: dict[int, Any] = {}
    assistant_indices = [
        index for index, message in enumerate(block) if message.role == "assistant"
    ]
    if not assistant_indices:
        return reasons + ["missing_assistant_action"]

    for index in assistant_indices:
        content = block[index].content.strip()
        try:
            content = _normalize_assistant_action(content)
            value = json.loads(content)
            if not isinstance(value, dict):
                raise TypeError("assistant action is not a JSON object")
            action = parse_action(content)
            parsed_actions[index] = action
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(
                f"invalid_assistant_json[{index}]:{type(exc).__name__}:{exc}"
            )
            continue
        if action.kind == "tool" and (
            index + 1 >= len(block)
            or block[index + 1].role != "user"
            or not block[index + 1].content.lstrip().startswith("<tool_result>")
        ):
            reasons.append(f"tool_without_result[{index}]")

    control_index = assistant_indices[-1]
    control_action = parsed_actions.get(control_index)
    if control_action is None or control_action.kind != "control":
        reasons.append("last_assistant_is_not_terminal_control")
        return reasons
    if control_index != len(block) - 1:
        reasons.append("messages_after_terminal_control")
    if any(
        action.kind == "control"
        for index, action in parsed_actions.items()
        if index != control_index
    ):
        reasons.append("early_terminal_control")

    try:
        control_content = _normalize_assistant_action(
            block[control_index].content.strip()
        )
        control_value = json.loads(control_content)
        answer = control_value.get("answer")
        decision_value = (
            answer if isinstance(answer, dict) else parse_json_object(str(answer))
        )
        ManagerDecision.from_dict(decision_value)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        reasons.append(f"invalid_manager_decision:{type(exc).__name__}:{exc}")
    return reasons


def _normalize_assistant_action(content: str) -> str:
    """Return the JSON payload from an optional exact tool-call wrapper.

    A Manager often narrates a sentence before its action object. The runtime
    accepts that through ``parse_json_object`` and executes the action, so the
    step really happened; rejecting it here would discard a valid decision over
    a prefix the runtime already ignored. Recover it the same way the runtime
    does, and hand back the serialized object rather than the original text so
    the exported record teaches the required action shape rather than the prose.
    """
    stripped = content.strip()
    opening = "<tool_call>"
    closing = "</tool_call>"
    wrapped = stripped.startswith(opening)
    payload = stripped[len(opening) :].strip() if wrapped else stripped
    if wrapped and payload.endswith(closing):
        payload = payload[: -len(closing)].strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        value = parse_json_object(payload)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if not isinstance(value, dict):
        raise TypeError("assistant action is not a JSON object")
    if wrapped:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return content


def _content_for_export(message: Message) -> str:
    if message.role != "assistant":
        return message.content
    try:
        return _normalize_assistant_action(message.content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return message.content


def _activation_coordinates(
    blocks: list[list[Message]],
) -> list[tuple[str, int]]:
    coordinates: list[tuple[str, int]] = []
    cycle_event = ""
    action_index = 0
    for block in blocks:
        event_type = str(block[0].metadata.get("event_type", ""))
        if event_type != "ACTION_RESULT":
            cycle_event = event_type
            action_index = 1
        else:
            action_index = max(1, action_index + 1)
        coordinates.append((cycle_event or event_type, action_index))
    return coordinates


def _assign_failures(
    coordinates: list[tuple[str, int]],
    validations: list[list[str]],
    task_ids: list[str],
    failures: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    assigned: dict[int, dict[str, Any]] = {}
    unmatched: list[dict[str, Any]] = []
    for failure in failures:
        key = (
            str(failure.get("event_type", "")),
            int(failure.get("action_index") or 0),
        )
        failure_task_id = str(failure.get("task_id", ""))
        candidates = [
            index
            for index, coordinate in enumerate(coordinates)
            if coordinate == key
            and index not in assigned
            and (not failure_task_id or task_ids[index] == failure_task_id)
        ]
        invalid = [index for index in candidates if validations[index]]
        if invalid:
            assigned[invalid[0]] = failure
        elif len(candidates) == 1:
            assigned[candidates[0]] = failure
        else:
            unmatched.append(failure)
    return assigned, unmatched
