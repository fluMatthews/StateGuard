from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import smolagents.agents as agents_module
    import smolagents.local_python_executor as executor_module
    from stateguard.adapters.dabstep.runtime_compat import (
        COMPAT_RUNTIME_PROFILE,
        apply_smolagents_runtime_fixes,
        configure_smolagents_runtime,
    )
    from stateguard.adapters.dabstep.executor import DABstepProbeExecutor
    from stateguard.adapters.dabstep.workspace import DABstepWorkspace
except ImportError:  # Core tests do not require DABstep extras.
    agents_module = None
    executor_module = None


@unittest.skipIf(agents_module is None, "DABstep optional dependencies are not installed")
class RuntimeCompatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = apply_smolagents_runtime_fixes(max_operations=20_000_000)

    def execute(self, code: str):
        state = {}
        output, is_final = executor_module.evaluate_python_code(
            code,
            static_tools=executor_module.BASE_PYTHON_TOOLS.copy(),
            state=state,
        )
        return state, output, is_final

    def test_continue_skips_rest_of_current_iteration(self):
        state, _, _ = self.execute(
            "kept = []\n"
            "for value in range(5):\n"
            "    if value % 2 == 0:\n"
            "        continue\n"
            "    kept.append(value)\n"
        )
        self.assertEqual(state["kept"], [1, 3])

    def test_break_and_for_else_match_python(self):
        state, _, _ = self.execute(
            "seen = []\n"
            "for value in range(5):\n"
            "    if value == 2:\n"
            "        break\n"
            "    seen.append(value)\n"
            "else:\n"
            "    seen.append(99)\n"
            "completed = []\n"
            "for value in range(2):\n"
            "    completed.append(value)\n"
            "else:\n"
            "    completed.append(99)\n"
        )
        self.assertEqual(state["seen"], [0, 1])
        self.assertEqual(state["completed"], [0, 1, 99])

    def test_next_accepts_generator_expression(self):
        state, _, _ = self.execute(
            "first = next(value for value in range(6) if value > 3)\n"
        )
        self.assertEqual(state["first"], 4)

    def test_set_comprehension_is_supported(self):
        state, _, _ = self.execute(
            "unique = {value % 3 for value in range(7)}\n"
        )
        self.assertEqual(state["unique"], {0, 1, 2})

    def test_repr_builtin_is_available(self):
        tools = executor_module.BASE_PYTHON_TOOLS.copy()
        state = {}
        executor_module.evaluate_python_code(
            "shown = repr({'x': 1})", static_tools=tools, state=state
        )
        self.assertEqual(state["shown"], "{'x': 1}")

    def test_complete_code_with_missing_end_fence_is_recovered(self):
        parsed = agents_module.parse_code_blobs(
            "Thought: compute\nCode:\n```py\nanswer = 40 + 2"
        )
        self.assertEqual(parsed, "answer = 40 + 2")

    def test_incomplete_python_is_not_silently_salvaged(self):
        with self.assertRaises(ValueError):
            agents_module.parse_code_blobs(
                "Thought: compute\nCode:\n```py\nanswer = (40 +"
            )

    def test_incomplete_last_block_does_not_execute_earlier_block(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            agents_module.parse_code_blobs(
                "Code:\n```py\nsetup = 1\n```\n"
                "Code:\n```py\nanswer = (setup +"
            )

    def test_complete_unclosed_last_block_is_joined_with_earlier_blocks(self):
        parsed = agents_module.parse_code_blobs(
            "Code:\n```py\nsetup = 40\n```\n"
            "Code:\n```py\nanswer = setup + 2"
        )
        self.assertEqual(parsed, "setup = 40\n\nanswer = setup + 2")

    def test_operation_limit_is_configurable(self):
        self.assertEqual(executor_module.MAX_OPERATIONS, 20_000_000)
        self.assertEqual(
            self.report["patch_version"], "dabstep-smolagents-1.3-compat-v1"
        )

    def test_explicit_profile_report_is_auditable(self):
        report = configure_smolagents_runtime(
            COMPAT_RUNTIME_PROFILE, max_operations=20_000_000
        )
        self.assertEqual(report["runtime_profile"], "compat-v1")
        self.assertEqual(report["smolagents_version"], "1.3.0")
        self.assertIn("for-loop break/continue semantics", report["fixes"])

    def test_probe_uses_the_same_compat_interpreter_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.csv").write_text("x\n1\n", encoding="utf-8")
            workspace = DABstepWorkspace(root)
            interpreter = executor_module.LocalPythonInterpreter([], tools={})
            worker = SimpleNamespace(
                native_agent=SimpleNamespace(python_executor=interpreter)
            )
            workspace.bind_worker(worker)
            result = DABstepProbeExecutor(
                workspace, runtime_profile=COMPAT_RUNTIME_PROFILE
            ).execute(
                "kept=[]\n"
                "for value in range(5):\n"
                "    if value % 2 == 0:\n"
                "        continue\n"
                "    kept.append(value)\n"
                "unique={value % 3 for value in range(7)}\n"
                "first=next(value for value in range(6) if value > 3)\n"
                "print(kept, sorted(unique), first, repr(dict(x=1)))"
            )
            self.assertTrue(result.ok, result.error)
            self.assertIn("[1, 3] [0, 1, 2] 4", result.stdout)


if __name__ == "__main__":
    unittest.main()
