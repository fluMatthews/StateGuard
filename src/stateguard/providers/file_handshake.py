from __future__ import annotations

import json
import os
import tempfile
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from stateguard.core.models import Message

from .base import ModelResponse


class FileHandshakeModelClient:
    """Bridge a live external model into the normal ``ModelClient`` contract.

    Each completion request is persisted for audit and the caller blocks until
    the matching response is atomically published.  The external model may
    return either a ``{"content": "..."}`` envelope or one raw ReAct action
    object.  Nothing in the Manager, tools, harness, state lifecycle, or repair
    controller is special-cased for this bridge.

    This is a transport adapter for attaching a live external model process; it
    is not a StateGuard method action.  Response producers should call
    :func:`publish_response` (or this module's CLI) instead of writing the final
    response path directly.
    """

    def __init__(
        self,
        exchange_dir: Path,
        *,
        poll_seconds: float = 1.0,
        timeout_seconds: float = 7200.0,
        invalid_response_grace_seconds: float = 10.0,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if invalid_response_grace_seconds <= 0:
            raise ValueError("invalid_response_grace_seconds must be positive")
        self.exchange_dir = exchange_dir.expanduser().resolve()
        self.exchange_dir.mkdir(parents=True, exist_ok=True)
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.invalid_response_grace_seconds = invalid_response_grace_seconds
        self.sequence = 0

    def complete(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        self.sequence += 1
        stem = f"completion_{self.sequence:06d}"
        request_path = self.exchange_dir / f"{stem}.request.json"
        response_path = self.exchange_dir / f"{stem}.response.json"
        if response_path.exists():
            response_path.unlink()
        _atomic_write_json(
            request_path,
            {
                "protocol": "stateguard-file-model-v1",
                "sequence": self.sequence,
                "messages": [
                    {
                        key: value
                        for key, value in {
                            "role": message.role,
                            "content": message.content,
                            "name": message.name,
                            "metadata": message.metadata or None,
                        }.items()
                        if value is not None
                    }
                    for message in messages
                ],
                "tools": tools,
            },
        )
        print(
            f"PAUSED external Manager completion {self.sequence}; "
            f"waiting for {response_path}",
            flush=True,
        )
        started = time.monotonic()
        invalid_since: float | None = None
        invalid_fingerprint: tuple[int, int] | None = None
        while True:
            if time.monotonic() - started > self.timeout_seconds:
                raise TimeoutError(
                    f"external Manager completion {self.sequence} timed out"
                )
            if response_path.is_file():
                try:
                    value: Any = json.loads(response_path.read_text(encoding="utf-8"))
                    break
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    # A non-atomic producer may briefly expose an empty/partial
                    # file. Do not turn that transport race into a Manager
                    # action failure. A stable malformed file still fails after
                    # a short grace period with a useful error.
                    try:
                        stat = response_path.stat()
                        fingerprint = (stat.st_size, stat.st_mtime_ns)
                    except OSError:
                        fingerprint = None
                    now = time.monotonic()
                    if fingerprint != invalid_fingerprint:
                        invalid_fingerprint = fingerprint
                        invalid_since = now
                    elif (
                        invalid_since is not None
                        and now - invalid_since > self.invalid_response_grace_seconds
                    ):
                        raise ValueError(
                            f"invalid external Manager response {response_path}: {exc}"
                        ) from exc
            time.sleep(self.poll_seconds)
        if isinstance(value, dict) and set(value).issubset(
            {"content", "reasoning", "metadata"}
        ) and "content" in value:
            content = value["content"]
            if not isinstance(content, str):
                raise ValueError("external Manager response.content must be a string")
            reasoning = value.get("reasoning", "")
            metadata = value.get("metadata")
            if not isinstance(reasoning, str):
                raise ValueError("external Manager response.reasoning must be a string")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("external Manager response.metadata must be an object")
            return ModelResponse(content, reasoning=reasoning, metadata=metadata)
        if not isinstance(value, dict):
            raise ValueError(
                "external Manager response must be a ReAct action object or content envelope"
            )
        return ModelResponse(json.dumps(value, ensure_ascii=False))


def _atomic_write_json(path: Path, value: Any) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def publish_response(exchange_dir: Path, sequence: int, value: Any) -> Path:
    """Validate and atomically publish one external Manager response."""
    if sequence < 1:
        raise ValueError("sequence must be positive")
    if not isinstance(value, dict):
        raise ValueError("external Manager response must be a JSON object")
    directory = exchange_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    response_path = directory / f"completion_{sequence:06d}.response.json"
    _atomic_write_json(response_path, value)
    return response_path


def _main() -> int:
    parser = ArgumentParser(
        description="Atomically publish a StateGuard file-handshake response"
    )
    parser.add_argument("--exchange-dir", type=Path, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--response-file", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.response_file.read_text(encoding="utf-8"))
    path = publish_response(args.exchange_dir, args.sequence, value)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
