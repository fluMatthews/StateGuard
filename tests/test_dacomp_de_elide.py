"""One DE step can exceed the Manager's whole window on its own.

Halving the pending trace cannot fix that: when a single step is all that is
left, dropping it leaves the Manager reviewing nothing. These tests pin the
escape hatch -- the Worker shortens that step, the buffer keeps the rewrite, and
Workers without the hook keep the drop-only behaviour.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from stateguard.adapters.dacomp.worker_de import _REASONING_HEAD, _REASONING_TAIL, DACompDEWorkerAgent
from stateguard.core.events import ReActStep
from stateguard.core.models import AgentAction, ToolResult
from stateguard.runtime.trace import TraceBuffer, TraceStatus


def _step(step_id: int, command: str, output: str, reasoning: str = "why") -> ReActStep:
    return ReActStep(
        step_id=step_id,
        action=AgentAction(
            kind="tool",
            reasoning=reasoning,
            tool_name="execute_bash",
            arguments={"command": command},
        ),
        observation=ToolResult("execute_bash", True, output, {}, None),
        done=False,
        raw_model_output="raw",
        metadata={"official_action": command, "official_observation": output},
    )


def _agent() -> DACompDEWorkerAgent:
    return object.__new__(DACompDEWorkerAgent)


class DEElideStepTests(unittest.TestCase):
    def test_short_step_is_left_alone(self):
        self.assertIsNone(_agent().elide_step(_step(1, "ls -la", "two files")))

    def test_long_command_keeps_head_and_tail(self):
        # The tail carries the step's own verification -- a closing print or ls.
        command = "python3 - <<'PY'\n" + "x = 1\n" * 6000 + "print('wrote', 12, 'files')\nPY"
        shortened = _agent().elide_step(_step(1, command, "ok"))
        assert shortened is not None
        rewritten = shortened.action.arguments["command"]
        self.assertLess(len(rewritten), len(command))
        self.assertIn("python3 - <<'PY'", rewritten)
        self.assertIn("print('wrote', 12, 'files')", rewritten)
        self.assertIn("characters elided", rewritten)

    def test_long_output_keeps_head_and_tail(self):
        output = "--- config\n" + "row\n" * 6000 + "EXIT:0"
        shortened = _agent().elide_step(_step(1, "ls", output))
        assert shortened is not None
        self.assertIn("--- config", shortened.observation.output)
        self.assertIn("EXIT:0", shortened.observation.output)
        self.assertIn("characters elided", shortened.observation.output)

    def test_long_reasoning_is_shortened_head_and_tail(self):
        # The Worker's chain of thought was the one field elision skipped, and
        # it is the largest: 856 characters at the median across 763 DE steps
        # but 62,220 at most. One review overflowed on a block whose reasoning
        # was 95% of 141,151 characters.
        reasoning = "open " * 1200 + "MIDDLE " * 4000 + " close" * 800
        shortened = _agent().elide_step(_step(1, "x", "y", reasoning))
        assert shortened is not None
        kept = shortened.action.reasoning
        self.assertLess(len(kept), len(reasoning))
        self.assertLessEqual(len(kept), _REASONING_HEAD + _REASONING_TAIL + 60)
        self.assertTrue(kept.startswith(reasoning[:_REASONING_HEAD]))
        self.assertTrue(kept.endswith(reasoning[-_REASONING_TAIL:]))
        self.assertIn("characters elided", kept)

    def test_short_reasoning_is_left_alone(self):
        # 90.4% of DE steps sit under the cap and must pass through untouched.
        reasoning = "a" * (_REASONING_HEAD + _REASONING_TAIL)
        shortened = _agent().elide_step(_step(1, "x" * 9000, "y", reasoning))
        assert shortened is not None
        self.assertEqual(shortened.action.reasoning, reasoning)

    def test_a_step_only_over_on_reasoning_still_reports_a_change(self):
        # Command and output within budget must not mask an oversized thought.
        step = _step(1, "ls", "ok", "z" * 40_000)
        shortened = _agent().elide_step(step)
        self.assertIsNotNone(shortened)
        self.assertLess(len(shortened.action.reasoning), 40_000)

    def test_audit_metadata_keeps_the_full_text(self):
        command = "a" * 9000
        shortened = _agent().elide_step(_step(1, command, "b" * 9000))
        assert shortened is not None
        self.assertEqual(shortened.metadata["official_action"], command)

    def test_a_second_pass_reports_nothing_left_to_shrink(self):
        agent = _agent()
        once = agent.elide_step(_step(1, "a" * 9000, "b" * 9000))
        assert once is not None
        self.assertIsNone(agent.elide_step(once))

    def test_failed_step_error_is_shortened_with_its_output(self):
        output = "boom\n" * 3000
        step = _step(1, "ls", output)
        step = replace(
            step,
            observation=ToolResult("execute_bash", False, output, {}, output),
        )
        shortened = _agent().elide_step(step)
        assert shortened is not None
        self.assertEqual(shortened.observation.error, shortened.observation.output)


class TraceBufferReplaceTests(unittest.TestCase):
    def setUp(self):
        self.buffer = TraceBuffer()
        self.buffer.start_unit("unit")
        self.buffer.append(_step(1, "ls", "ok"))
        self.buffer.append(_step(2, "ls", "ok"))

    def test_replacement_is_permanent(self):
        rewritten = _step(2, "shortened", "shortened")
        self.buffer.replace_pending_step(rewritten)
        self.assertEqual(
            [s.action.arguments["command"] for s in self.buffer.steps], ["ls", "shortened"]
        )

    def test_unknown_step_is_rejected(self):
        with self.assertRaises(KeyError):
            self.buffer.replace_pending_step(_step(99, "x", "y"))

    def test_settled_step_is_rejected(self):
        self.buffer.records[0].status = TraceStatus.PASSED
        with self.assertRaises(ValueError):
            self.buffer.replace_pending_step(_step(1, "x", "y"))


if __name__ == "__main__":
    unittest.main()


class ElideBeforeDropTests(unittest.TestCase):
    """The retry loop shortens every pending step before it drops any.

    Dropping is the lossier move: a DE step can overflow the window on its own,
    so halving runs all the way down to one step and a review of one step forms
    no state. Measured on impl-010, seven pending steps are 51,482 characters
    raw against a 41,073-character budget but 33,273 once shortened -- the whole
    window survives instead of six sevenths of it being thrown away.
    """

    def setUp(self):
        self.worker = _agent()
        self.buffer = TraceBuffer()
        self.buffer.start_unit("unit")
        for i in range(1, 8):
            self.buffer.append(_step(i, "python3 -c '" + "x=1;" * 3000 + "'", "row\n" * 3000))

    def _elide_all(self):
        elided = []
        for step in self.buffer.steps:
            shortened = self.worker.elide_step(step)
            if shortened is None:
                continue
            self.buffer.replace_pending_step(shortened)
            elided.append(step.step_id)
        return elided

    def _size(self):
        return sum(
            len(s.action.arguments["command"]) + len(s.observation.output)
            for s in self.buffer.steps
        )

    def test_every_step_is_shortened_and_none_dropped(self):
        before = self._size()
        elided = self._elide_all()
        self.assertEqual(elided, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(self.buffer.steps), 7)
        self.assertLess(self._size(), before / 2)

    def test_a_second_pass_finds_nothing_so_the_loop_can_fall_through(self):
        self._elide_all()
        self.assertEqual(self._elide_all(), [])

    def test_steps_stay_shortened_for_later_reviews(self):
        self._elide_all()
        size = self._size()
        self.assertEqual(self._size(), size)
        self.assertTrue(
            all("characters elided" in s.action.arguments["command"] for s in self.buffer.steps)
        )
