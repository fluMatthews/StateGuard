from __future__ import annotations

import json
import threading
import time

from stateguard.core.models import Message
from stateguard.providers.file_handshake import (
    FileHandshakeModelClient,
    publish_response,
)


def test_file_handshake_accepts_raw_action(tmp_path) -> None:
    client = FileHandshakeModelClient(
        tmp_path, poll_seconds=0.01, timeout_seconds=2.0
    )

    def answer() -> None:
        request = tmp_path / "completion_000001.request.json"
        while not request.exists():
            time.sleep(0.01)
        payload = json.loads(request.read_text(encoding="utf-8"))
        assert payload["messages"][0]["content"] == "system"
        publish_response(
            tmp_path,
            1,
            {
                "type": "control",
                "reasoning": "safe to continue",
                "answer": {"action": "RESUME_WORKER"},
            },
        )

    thread = threading.Thread(target=answer)
    thread.start()
    response = client.complete(
        [Message("system", "system"), Message("user", "observation")], []
    )
    thread.join()
    assert json.loads(response.content)["answer"]["action"] == "RESUME_WORKER"


def test_file_handshake_waits_for_partial_response_to_finish(tmp_path) -> None:
    client = FileHandshakeModelClient(
        tmp_path,
        poll_seconds=0.01,
        timeout_seconds=2.0,
        invalid_response_grace_seconds=0.5,
    )

    def answer() -> None:
        request = tmp_path / "completion_000001.request.json"
        while not request.exists():
            time.sleep(0.01)
        response = tmp_path / "completion_000001.response.json"
        response.write_text("", encoding="utf-8")
        time.sleep(0.05)
        publish_response(
            tmp_path,
            1,
            {
                "type": "control",
                "reasoning": "safe to continue",
                "answer": {"action": "RESUME_WORKER"},
            },
        )

    thread = threading.Thread(target=answer)
    thread.start()
    response = client.complete([Message("system", "system")], [])
    thread.join()
    assert json.loads(response.content)["answer"]["action"] == "RESUME_WORKER"
