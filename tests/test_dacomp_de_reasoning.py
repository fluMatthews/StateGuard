"""The DE Worker must hand the Manager the same stated intent the other flows do.

DA and DABstep steps carry the Worker's reasoning; DE steps arrived empty because
OpenHands reads only the assistant message content, and a reasoning model that
answers with a tool call leaves that content empty. These tests pin the recovery
and, just as importantly, pin that ``action.thought`` -- the field that feeds the
Worker's own conversation -- is never written.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from stateguard.adapters.dacomp.worker_de_controller import _step_reasoning


@dataclass
class FakeToolCallMetadata:
    model_response: Any = None


@dataclass
class FakeAction:
    thought: str = ""
    tool_call_metadata: Any = None
    command: str = "ls -la"


def _response(**message: Any) -> FakeToolCallMetadata:
    return FakeToolCallMetadata({"choices": [{"message": message}]})


class DEStepReasoningTests(unittest.TestCase):
    def test_existing_thought_wins(self):
        action = FakeAction(
            thought="Inspect the contract first.",
            tool_call_metadata=_response(content="", reasoning_content="something else"),
        )
        self.assertEqual(_step_reasoning(action), "Inspect the contract first.")

    def test_reasoning_content_recovered_when_thought_is_empty(self):
        # The exact shape deepseek-v4-pro returns with a tool call.
        action = FakeAction(
            tool_call_metadata=_response(
                content="",
                reasoning_content="We need list files in current directory.",
                tool_calls=[{"function": {"name": "execute_bash"}}],
            )
        )
        self.assertEqual(
            _step_reasoning(action), "We need list files in current directory."
        )

    def test_worker_thought_is_never_written(self):
        action = FakeAction(tool_call_metadata=_response(reasoning_content="recovered"))
        _step_reasoning(action)
        # conversation_memory rebuilds the Worker's own history from this field.
        self.assertEqual(action.thought, "")

    def test_missing_metadata_is_not_an_error(self):
        self.assertEqual(_step_reasoning(FakeAction()), "")

    def test_malformed_response_is_not_an_error(self):
        for broken in (None, {}, {"choices": []}, {"choices": [{}]}, "not-a-dict"):
            self.assertEqual(
                _step_reasoning(FakeAction(tool_call_metadata=FakeToolCallMetadata(broken))),
                "",
            )

    def test_empty_reasoning_content_yields_empty_string(self):
        action = FakeAction(tool_call_metadata=_response(content="", reasoning_content=None))
        self.assertEqual(_step_reasoning(action), "")

    def test_object_message_is_read_through_its_attributes(self):
        class Message:
            def __init__(self):
                self.reasoning_content = "attribute form"

        meta = FakeToolCallMetadata({"choices": [{"message": Message()}]})
        self.assertEqual(_step_reasoning(FakeAction(tool_call_metadata=meta)), "attribute form")


if __name__ == "__main__":
    unittest.main()
