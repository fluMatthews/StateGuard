import json
import unittest

from stateguard.agents.worker import WorkerAgent
from stateguard.core.models import ToolResult
from stateguard.providers.base import ScriptedModelClient
from stateguard.runtime.executors import TrustedPythonExecutor
from stateguard.runtime.tools import FunctionTool, ToolRegistry
from stateguard.runtime.workspace import InMemoryWorkspace


class ReActAgentTest(unittest.TestCase):
    def test_worker_executes_tool_before_final_answer(self):
        workspace = InMemoryWorkspace()
        executor = TrustedPythonExecutor(workspace)

        def execute_python(code: str):
            result = executor.execute(code)
            return ToolResult("python", result.ok, result.stdout, error=result.error)

        tools = ToolRegistry([FunctionTool("python", "Execute trusted Python", execute_python)])
        model = ScriptedModelClient(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "reasoning": "Compute the requested value.",
                        "tool": "python",
                        "arguments": {"code": "result = 6 * 7\nprint(result)"},
                        "metadata": {"important_result": True},
                    }
                ),
                json.dumps({"type": "final", "answer": "42", "reasoning": "Use executed result."}),
            ]
        )
        worker = WorkerAgent(model, tools)
        worker.start("Compute 6 * 7")
        first = worker.step()
        second = worker.step()

        self.assertTrue(first.observation.ok)
        self.assertEqual(workspace.variables["result"], 42)
        self.assertTrue(second.done)
        self.assertEqual(worker.final_answer, "42")


if __name__ == "__main__":
    unittest.main()
