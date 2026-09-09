from __future__ import annotations

import json
import unittest
from unittest.mock import patch

try:
    from stateguard.adapters.dabstep.anthropic import AnthropicMessagesModel
except ImportError:
    AnthropicMessagesModel = None


class _Response:
    def __init__(self, blocks):
        self.blocks = blocks

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "content": self.blocks,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        ).encode()


@unittest.skipIf(AnthropicMessagesModel is None, "DABstep extras are not installed")
class DABstepAnthropicTests(unittest.TestCase):
    def _model(self):
        return AnthropicMessagesModel(
            model_id="anthropic/claude-sonnet-5",
            api_base="https://example.test",
            api_key="secret",
        )

    def test_explicitly_disables_hosted_tools_and_returns_text(self):
        model = self._model()
        response_body = _Response(
            [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": "42"},
            ]
        )
        with patch("urllib.request.urlopen", return_value=response_body) as urlopen:
            response = model(
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "question"},
                ],
                stop_sequences=["<end_code>"],
            )

        payload = json.loads(urlopen.call_args.args[0].data.decode())
        self.assertEqual(payload["model"], "claude-sonnet-5")
        self.assertEqual(payload["tools"], [])
        self.assertEqual(payload["system"], "system")
        self.assertEqual(response.content, "42")

    def test_converts_injected_python_tool_to_official_text_action(self):
        model = self._model()
        body = _Response(
            [
                {
                    "type": "tool_use",
                    "name": "python_interpreter",
                    "input": {"code": "print(42)"},
                }
            ]
        )
        with patch("urllib.request.urlopen", return_value=body) as urlopen:
            response = model([{"role": "user", "content": "compute"}])
        self.assertIn("Code:", response.content)
        self.assertIn("print(42)", response.content)
        self.assertEqual(urlopen.call_count, 1)

    def test_finalizer_retries_code_as_plain_text_without_extra_action(self):
        model = self._model()
        first = _Response([{"type": "text", "text": "Code:\n```py\nprint(42)\n```"}])
        second = _Response([{"type": "text", "text": "42"}])
        messages = [
            {
                "role": "system",
                "content": (
                    "An agent tried to answer a user query but it got stuck and failed "
                    "to do so. You are tasked with providing an answer instead."
                ),
            },
            {"role": "user", "content": "Based on the above, answer now."},
        ]
        with patch("urllib.request.urlopen", side_effect=[first, second]) as urlopen:
            response = model(messages)

        self.assertEqual(response.content, "42")
        self.assertEqual(urlopen.call_count, 2)
        retry_payload = json.loads(urlopen.call_args_list[1].args[0].data.decode())
        self.assertEqual(retry_payload["tools"], [])
        self.assertIn("Format correction only", retry_payload["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
