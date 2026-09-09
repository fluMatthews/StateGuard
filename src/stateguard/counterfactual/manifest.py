from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from stateguard.core.models import Message
from stateguard.sft.context import (
    DEFAULT_MANAGER_CONTEXT_CHARS,
    DEFAULT_MANAGER_OUTPUT_RESERVE_CHARS,
    select_manager_context,
)

from .hashing import file_digest, messages_digest, source_tree_digest
from .models import ManagerReplayCall, ReplayManifest, WorkerReplayCall


_PARENT_FILENAMES = (
    "trajectory.json",
    "manager_session.json",
    "manager_failures.json",
    "run_metadata.json",
    "sft_export_report.json",
    "stateguard/worker.jsonl",
)


def extract_replay_manifest(
    run_dir: Path,
    *,
    require_worker_replayable: bool = True,
    manager_max_context_chars: int | None = DEFAULT_MANAGER_CONTEXT_CHARS,
    manager_reserved_output_chars: int = DEFAULT_MANAGER_OUTPUT_RESERVE_CHARS,
) -> ReplayManifest:
    """Build a deterministic model-call tape from one completed Worker run."""
    run_dir = run_dir.expanduser().resolve(strict=True)
    trajectory = _read_object(run_dir / "trajectory.json")
    manager_session_path = run_dir / "manager_session.json"
    manager_session = (
        _read_object(manager_session_path)
        if manager_session_path.is_file()
        else {"messages": []}
    )
    run_metadata = _read_object(run_dir / "run_metadata.json")
    failures_path = run_dir / "manager_failures.json"
    manager_failures = json.loads(failures_path.read_text(encoding="utf-8")) if failures_path.is_file() else []
    if require_worker_replayable:
        _require_replayable_worker_parent(trajectory)

    source = str(trajectory.get("source") or run_metadata.get("source") or "")
    mode = str(trajectory.get("mode") or run_metadata.get("mode") or "")
    task_id = str(trajectory.get("task_id") or run_metadata.get("task_id") or "")
    if not source or not mode or not task_id:
        raise ValueError("parent run is missing source, mode, or task id")

    raw_worker_responses = _read_raw_worker_responses(run_dir, trajectory)
    worker_calls = _extract_worker_calls(trajectory, raw_worker_responses)
    manager_calls = _extract_manager_calls(
        manager_session,
        manager_max_context_chars,
        manager_reserved_output_chars,
    )
    if not worker_calls:
        raise ValueError("parent run contains no Worker model calls")
    manager_quality = _assess_manager_quality(
        run_dir, manager_failures, manager_session, manager_session_path.is_file()
    )

    parent_files = {
        name: file_digest(run_dir / name)
        for name in _PARENT_FILENAMES
        if (run_dir / name).is_file()
    }
    return ReplayManifest(
        parent_run=str(run_dir),
        source=source,
        mode=mode,
        task_id=task_id,
        worker_calls=worker_calls,
        manager_calls=manager_calls,
        parent_files=parent_files,
        metadata={
            "worker_model": trajectory.get("model") or run_metadata.get("model"),
            "worker_budget_per_unit": run_metadata.get("worker_budget_per_unit"),
            "review_cadence": run_metadata.get("review_cadence"),
            "manager_max_context_chars": manager_max_context_chars,
            "manager_reserved_output_chars": manager_reserved_output_chars,
            "worker_replayable_required": require_worker_replayable,
            "manager_quality": manager_quality,
            "runtime_fingerprint": source_tree_digest(
                Path(__file__).resolve().parents[1]
            ),
            "worker_response_source": (
                "stateguard/worker.jsonl.raw_model_output"
                if raw_worker_responses is not None
                else "trajectory.assistant.content"
            ),
        },
    )


def write_replay_manifest(path: Path, manifest: ReplayManifest) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_replay_manifest(path: Path) -> ReplayManifest:
    return ReplayManifest.from_dict(_read_object(path.expanduser().resolve(strict=True)))


def verify_parent_artifacts(manifest: ReplayManifest) -> None:
    """Reject a replay if any source artifact changed after manifest creation."""
    parent = Path(manifest.parent_run).resolve(strict=True)
    for name, expected in manifest.parent_files.items():
        path = parent / name
        if not path.is_file():
            raise FileNotFoundError(f"replay parent artifact disappeared: {path}")
        actual = file_digest(path)
        if actual != expected:
            raise ValueError(f"replay parent artifact changed: {path}")


def _require_replayable_worker_parent(trajectory: dict[str, Any]) -> None:
    """Require a recorded Worker tape; later unit outcomes are orthogonal.

    A counterfactual branch switches to live execution at one concrete Worker
    call. A later exhausted or failed unit therefore cannot invalidate an
    earlier replayable prefix. Target existence is checked separately by
    ``ReplayManifest.worker_call`` when the intervention is loaded.
    """
    units = trajectory.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("Worker parent run contains no units")
    if not any(
        isinstance(row, dict) and isinstance(row.get("trajectory"), list)
        for row in units
    ):
        raise ValueError("Worker parent run contains no recorded unit trajectory")


def _assess_manager_quality(
    run_dir: Path,
    manager_failures: Any,
    manager_session: dict[str, Any],
    manager_session_available: bool,
) -> dict[str, Any]:
    """Record Manager issues for post-hoc filtering without blocking replay."""
    issues: list[str] = []
    if not manager_session_available:
        issues.append("manager_session_unavailable")
    if manager_failures:
        count = len(manager_failures) if isinstance(manager_failures, list) else 1
        issues.append(f"runtime_failures:{count}")
    raw_messages = manager_session.get("messages")
    if not isinstance(raw_messages, list):
        issues.append("missing_or_invalid_session_messages")
        raw_messages = []
    for index, row in enumerate(raw_messages):
        if not isinstance(row, dict):
            issues.append(f"invalid_message:{index}")
            continue
        content = str(row.get("content", "")).strip()
        if content.startswith("<manager_protocol_error>"):
            issues.append(f"protocol_retry:{index}")
        if content.startswith("<tool_result>"):
            payload = content.removeprefix("<tool_result>").removesuffix(
                "</tool_result>"
            ).strip()
            try:
                value = json.loads(payload)
            except json.JSONDecodeError:
                issues.append(f"invalid_tool_result:{index}")
            else:
                if isinstance(value, dict) and value.get("ok") is False:
                    issues.append(f"failed_tool_action:{index}")

    path = run_dir / "sft_export_report.json"
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            issues.append("invalid_sft_export_report")
        else:
            if not isinstance(value, list):
                issues.append("invalid_sft_export_report")
            else:
                rejected = [
                    row
                    for row in value
                    if not isinstance(row, dict) or not row.get("accepted")
                ]
                if rejected:
                    issues.append(f"rejected_sft_activations:{len(rejected)}")
    return {
        "available": manager_session_available,
        "clean": manager_session_available and not issues,
        "issues": issues,
    }


def _extract_worker_calls(
    trajectory: dict[str, Any], raw_responses: tuple[str, ...] | None = None
) -> tuple[WorkerReplayCall, ...]:
    conversation: list[dict[str, str]] = []
    calls: list[WorkerReplayCall] = []
    units = trajectory.get("units")
    if not isinstance(units, list):
        raise TypeError("trajectory units must be a list")
    for unit in units:
        if not isinstance(unit, dict):
            raise TypeError("trajectory unit must be an object")
        unit_id = str(unit.get("unit_id", ""))
        rows = unit.get("trajectory")
        if not unit_id or not isinstance(rows, list):
            raise ValueError("trajectory unit is missing unit_id or trajectory")
        step_id = 0
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("Worker trajectory message must be an object")
            role = str(row.get("role", ""))
            content = row.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise ValueError(f"invalid Worker message in {unit_id}: {row!r}")
            if role == "assistant":
                step_id += 1
                response = content
                if raw_responses is not None:
                    raw_index = len(calls)
                    if raw_index >= len(raw_responses):
                        raise ValueError("Worker raw-response log is shorter than trajectory")
                    response = raw_responses[raw_index]
                calls.append(
                    WorkerReplayCall(
                        call_index=len(calls) + 1,
                        unit_id=unit_id,
                        step_id=step_id,
                        input_hash=messages_digest(conversation),
                        response=response,
                    )
                )
            conversation.append({"role": role, "content": content})
        expected_steps = int(unit.get("worker_steps", step_id))
        if step_id != expected_steps:
            raise ValueError(
                f"Worker call count mismatch for {unit_id}: {step_id} != {expected_steps}"
            )
    final_conversation = trajectory.get("conversation")
    if not isinstance(final_conversation, list):
        raise TypeError("trajectory conversation must be a list")
    if [dict(row) for row in final_conversation] != conversation:
        raise ValueError("unit trajectories do not reconstruct the global Worker conversation")
    if raw_responses is not None and len(raw_responses) != len(calls):
        raise ValueError(
            "Worker raw-response log length does not match trajectory calls: "
            f"{len(raw_responses)} != {len(calls)}"
        )
    return tuple(calls)


def _read_raw_worker_responses(
    run_dir: Path, trajectory: dict[str, Any]
) -> tuple[str, ...] | None:
    path = run_dir / "stateguard" / "worker.jsonl"
    if not path.is_file():
        return None
    records: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(
            value.get("raw_model_output"), str
        ):
            raise ValueError(f"invalid Worker raw-response record at {path}:{line_number}")
        metadata = value.get("metadata") or {}
        official_action = (
            str(metadata.get("official_action", ""))
            if isinstance(metadata, dict)
            else ""
        )
        records.append((value["raw_model_output"], official_action))

    assistant_actions = [
        str(message["content"])
        for unit in trajectory.get("units", ())
        if isinstance(unit, dict)
        for message in unit.get("trajectory", ())
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    responses: list[str] = []
    cursor = 0
    for action in assistant_actions:
        while cursor < len(records):
            raw, official = records[cursor]
            cursor += 1
            if action == official or action == raw:
                responses.append(raw)
                break
        else:
            raise ValueError(
                "Worker raw-response log cannot reconstruct the retained clean trajectory"
            )
    return tuple(responses)


def _extract_manager_calls(
    session: dict[str, Any],
    max_context_chars: int | None,
    reserved_output_chars: int,
) -> tuple[ManagerReplayCall, ...]:
    raw_messages = session.get("messages")
    if not isinstance(raw_messages, list):
        raise TypeError("manager_session messages must be a list")
    messages = [_manager_message(row) for row in raw_messages]
    calls: list[ManagerReplayCall] = []
    task_id = ""
    event_type = ""
    prefix: list[Message] = []
    for message in messages:
        if message.metadata.get("manager_block_start"):
            task_id = str(message.metadata.get("task_id", task_id))
            event_type = str(message.metadata.get("event_type", event_type))
        if message.role == "assistant":
            selected = select_manager_context(
                prefix, max_context_chars, reserved_output_chars
            )
            calls.append(
                ManagerReplayCall(
                    call_index=len(calls) + 1,
                    task_id=task_id,
                    event_type=event_type,
                    input_hash=messages_digest(selected),
                    response=message.content,
                    reasoning=str(message.metadata.get("reasoning", "")),
                )
            )
        prefix.append(message)
    return tuple(calls)


def _manager_message(value: Any) -> Message:
    if not isinstance(value, dict):
        raise TypeError("Manager message must be an object")
    metadata = value.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise TypeError("Manager message metadata must be an object")
    return Message(
        str(value.get("role", "")),
        str(value.get("content", "")),
        value.get("name"),
        dict(metadata),
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _main() -> int:
    parser = argparse.ArgumentParser(description="Extract a Worker replay manifest")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete-worker", action="store_true")
    args = parser.parse_args()
    manifest = extract_replay_manifest(
        args.run_dir,
        require_worker_replayable=not args.allow_incomplete_worker,
    )
    write_replay_manifest(args.output, manifest)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
