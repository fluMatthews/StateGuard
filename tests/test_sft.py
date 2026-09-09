from __future__ import annotations

import json
import unittest

from stateguard.core.models import Message
from stateguard.sft import (
    DEFAULT_MANAGER_CONTEXT_CHARS,
    DEFAULT_MANAGER_OUTPUT_RESERVE_CHARS,
    export_manager_activations,
    select_manager_context,
)


class SFTActivationExportTests(unittest.TestCase):
    def test_one_session_splits_into_one_record_per_manager_activation(self):
        records, report = _export(_session(_valid_blocks()))
        self.assertEqual(len(records), 3)
        self.assertEqual([row["accepted"] for row in report], [True] * 3)
        for index, record in enumerate(records):
            messages = record["messages"]
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[1]["role"], "user")
            self.assertEqual(messages[-1]["role"], "assistant")
            observations = [
                row["content"]
                for row in messages
                if row["content"].startswith("<manager_observation>")
            ]
            self.assertIn(f"observation {index + 1}", observations[-1])
            self.assertEqual(len(observations), index + 1)

    def test_structurally_invalid_activations_are_excluded_with_reasons(self):
        blocks = _valid_blocks()
        blocks.append(
            [
                _observation(4, "STEP_WINDOW"),
                Message("assistant", "I think the worker is fine, let us continue."),
            ]
        )
        blocks.append(
            [
                _observation(5, "STEP_WINDOW"),
                Message("assistant", _tool_call("load_state", {"state_id": "S1"})),
                Message("assistant", _control({"action": "RESUME_WORKER"})),
            ]
        )
        records, report = _export(_session(blocks))
        self.assertEqual(len(records), 3)
        rejected = [row for row in report if not row["accepted"]]
        self.assertEqual(len(rejected), 2)
        joined = " ".join(reason for row in rejected for reason in row["reasons"])
        self.assertIn("invalid_assistant_json", joined)
        self.assertIn("tool_without_result", joined)

    def test_final_manager_failure_rejects_only_its_matching_activation(self):
        failures = [
            {
                "event_type": "STEP_WINDOW",
                "action_index": 2,
                "error_type": "ActionParseError",
                "message": "invalid JSON",
            }
        ]
        records, report = _export(_session(_valid_blocks()), failures)
        self.assertEqual(len(records), 2)
        self.assertEqual([row["accepted"] for row in report], [True, False, True])
        self.assertIn("manager_failure:ActionParseError", report[1]["reasons"][0])

    def test_eligible_task_ids_filter_only_the_failed_turn(self):
        blocks = [
            [
                _observation(1, "TURN_END", task_id="turn-1"),
                Message("assistant", _control({"action": "RESUME_WORKER"})),
            ],
            [
                _observation(2, "TURN_END", task_id="turn-2"),
                Message("assistant", _control({"action": "RESUME_WORKER"})),
            ],
        ]
        records, report = export_manager_activations(
            _session(blocks),
            eligible_task_ids={"turn-1"},
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(report[0]["task_id"], "turn-1")
        self.assertTrue(report[0]["accepted"])
        self.assertEqual(report[1]["task_id"], "turn-2")
        self.assertFalse(report[1]["accepted"])
        self.assertIn("ineligible_task_unit", report[1]["reasons"])

    def test_exported_context_matches_the_runtime_selection(self):
        blocks = _valid_blocks()
        session = _session(blocks)
        records, _ = _export(session)
        messages = [
            Message(row["role"], row["content"], row.get("name"), row.get("metadata") or {})
            for row in session["messages"]
        ]
        prefix = messages[:2]
        cursor = 2
        for index, block in enumerate(blocks):
            cursor += len(block)
            expected = select_manager_context(
                messages[:cursor],
                DEFAULT_MANAGER_CONTEXT_CHARS,
                DEFAULT_MANAGER_OUTPUT_RESERVE_CHARS,
            )
            self.assertEqual(
                records[index]["messages"],
                [{"role": message.role, "content": message.content} for message in expected],
            )
            self.assertEqual(expected[:2], prefix)


    def test_exact_tool_call_wrapper_is_exported_as_plain_json(self):
        wrapped = (
            "<tool_call>\n"
            + _control({"action": "RESUME_WORKER"})
            + "\n</tool_call>"
        )
        blocks = [
            [
                _observation(1, "STEP_WINDOW"),
                Message("assistant", wrapped),
            ]
        ]

        records, report = _export(_session(blocks))

        self.assertEqual(len(records), 1)
        self.assertTrue(report[0]["accepted"])
        content = records[0]["messages"][-1]["content"]
        self.assertNotIn("<tool_call>", content)
        self.assertEqual(json.loads(content)["type"], "control")


def _export(session: dict, failures: list[dict] | None = None):
    return export_manager_activations(
        session,
        failures or [],
        max_context_chars=DEFAULT_MANAGER_CONTEXT_CHARS,
        reserved_output_chars=DEFAULT_MANAGER_OUTPUT_RESERVE_CHARS,
    )


def _session(blocks: list[list[Message]]) -> dict:
    messages = [
        Message("system", "manager system prompt"),
        Message("user", "controller\n\nTASK:\n{}"),
    ]
    for block in blocks:
        messages.extend(block)
    return {
        "messages": [
            {
                "role": message.role,
                "content": message.content,
                "name": message.name,
                "metadata": message.metadata,
            }
            for message in messages
        ]
    }


def _valid_blocks() -> list[list[Message]]:
    return [
        [
            _observation(1, "STEP_WINDOW"),
            Message(
                "assistant",
                _control(
                    {
                        "action": "OPEN_STATE",
                        "state_header": {
                            "id": "S1",
                            "constraints": [{"text": "Answer the question."}],
                        },
                    }
                ),
            ),
        ],
        [
            _observation(2, "ACTION_RESULT"),
            Message("assistant", _tool_call("load_state", {"state_id": "S1"})),
            Message("user", "<tool_result>\n{}\n</tool_result>"),
            Message(
                "assistant",
                _control(
                    {
                        "action": "UPDATE_STATE",
                        "state_update": {
                            "issue": "first window",
                            "used_variables": [],
                            "conclusions": ["ok"],
                            "source_interval": {"start": 1, "end": 1},
                        },
                    }
                ),
            ),
        ],
        [
            _observation(3, "ACTION_RESULT"),
            Message("assistant", _control({"action": "RESUME_WORKER"})),
        ],
    ]


def _observation(
    index: int, event_type: str, task_id: str = ""
) -> Message:
    return Message(
        "user",
        f"<manager_observation>\nobservation {index}\n</manager_observation>",
        None,
        {"manager_block_start": True, "event_type": event_type, "task_id": task_id},
    )


def _tool_call(tool: str, arguments: dict) -> str:
    return json.dumps(
        {"type": "tool", "reasoning": "check", "tool": tool, "arguments": arguments}
    )


def _control(action: dict) -> str:
    return json.dumps({"type": "control", "reasoning": "follow lifecycle", "answer": action})


if __name__ == "__main__":
    unittest.main()
