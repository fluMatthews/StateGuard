import unittest
from importlib.resources import files

from stateguard.adapters.dabstep.workflow import DABstepWorkflow
from stateguard.adapters.dacomp.workflow import DACompWorkflow
from stateguard.adapters.longds.workflow import LongDSWorkflow
from stateguard.agents.manager import render_manager_controller


def render(lifecycle: str) -> str:
    return render_manager_controller(lifecycle)


class ManagerPromptIsolationTests(unittest.TestCase):
    def test_controller_contains_no_flow_specific_examples(self):
        controller = (
            files("stateguard.prompts")
            .joinpath("manager_controller.txt")
            .read_text(encoding="utf-8")
        )
        self.assertNotIn("Pre-trace header", controller)
        self.assertNotIn("Deferred-content header", controller)
        self.assertNotIn('"mode":"confirm"', controller)
        self.assertNotIn('"mode":"reselect"', controller)
        self.assertNotIn('"mode":"select"', controller)

    def test_single_query_lifecycle_is_shared(self):
        self.assertEqual(DABstepWorkflow(5).lifecycle_prompt(), DACompWorkflow(5).lifecycle_prompt())

    def test_rendered_prompts_expose_only_the_current_relation_modes(self):
        longds = render(LongDSWorkflow({}, "system").lifecycle_prompt())
        dabstep = render(DABstepWorkflow().lifecycle_prompt())
        dacomp = render(DACompWorkflow().lifecycle_prompt())

        self.assertIn('"mode":"confirm"', longds)
        self.assertIn('"mode":"reselect"', longds)
        self.assertNotIn('"mode":"select"', longds)
        for prompt in (dabstep, dacomp):
            self.assertIn('"mode":"select"', prompt)
            self.assertNotIn('"mode":"confirm"', prompt)
            self.assertNotIn('"mode":"reselect"', prompt)


if __name__ == "__main__":
    unittest.main()
