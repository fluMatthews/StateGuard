"""A provider tool-call envelope that reaches parse_action as message content.

StateGuard_v1 is served with tool schemas, and a reasoning model sometimes
answers with the provider's own call shape written into content rather than into
tool_calls -- where OpenAICompatibleClient would already have converted it. Three
such responses appeared across the DA and DE runs; all three were complete calls
missing only the "type" the text protocol asks for.
"""

from __future__ import annotations

import unittest

from stateguard.agents.react import ActionParseError, parse_action


class NativeToolCallFallbackTests(unittest.TestCase):
    def test_the_shape_seen_in_the_de_runs(self):
        action = parse_action('{"name": "check_execution", "arguments": {"step_id": 49}}')
        self.assertEqual(action.kind, "tool")
        self.assertEqual(action.tool_name, "check_execution")
        self.assertEqual(action.arguments, {"step_id": 49})
        self.assertEqual(action.reasoning, "")

    def test_the_shape_seen_in_the_da_run(self):
        action = parse_action('{"name": "compile_python", "arguments": {"code": ""}}')
        self.assertEqual(action.tool_name, "compile_python")
        self.assertEqual(action.arguments, {"code": ""})

    def test_reasoning_is_carried_when_the_model_supplies_it(self):
        action = parse_action(
            '{"name": "load_state", "reasoning": "check S1", "arguments": {"state_id": "S1"}}'
        )
        self.assertEqual(action.reasoning, "check S1")

    def test_an_explicit_type_still_wins(self):
        # The text protocol stays authoritative when the model follows it.
        action = parse_action(
            '{"type": "tool", "tool": "run_probe", "name": "ignored",'
            ' "arguments": {"code": "1"}}'
        )
        self.assertEqual(action.tool_name, "run_probe")

    def test_a_name_without_arguments_is_still_rejected(self):
        with self.assertRaises(ActionParseError):
            parse_action('{"name": "check_execution"}')

    def test_a_blank_name_is_still_rejected(self):
        with self.assertRaises(ActionParseError):
            parse_action('{"name": "", "arguments": {}}')

    def test_non_object_arguments_are_still_rejected(self):
        with self.assertRaises(ActionParseError):
            parse_action('{"name": "check_execution", "arguments": [1, 2]}')

    def test_an_unrelated_object_is_still_rejected(self):
        with self.assertRaises(ActionParseError):
            parse_action('{"foo": "bar"}')

    def test_a_truncated_response_is_still_rejected(self):
        # The 25 of 28 malformed responses that were cut mid-reasoning must keep
        # failing: half an envelope is not an action.
        with self.assertRaises(ActionParseError):
            parse_action('{"type": "tool", "reasoning": "the worker has done 9 steps')


if __name__ == "__main__":
    unittest.main()
