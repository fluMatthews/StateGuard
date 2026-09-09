from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stateguard.adapters.dacomp.artifacts import write_da_stage1_artifacts
from stateguard.adapters.dacomp.dataset import DACompDataset, DACompTrack
from stateguard.adapters.dacomp.official_prompt import official_de_task_prompt

OFFICIAL_DACOMP_ROOT = Path("/fs/fast/u2024201619/DAComp-main")


@unittest.skipUnless(
    (OFFICIAL_DACOMP_ROOT / "methods" / "de-agent").is_dir(),
    "official DAComp repository is not present",
)
class DACompOfficialFidelityTests(unittest.TestCase):
    """The Worker prompt and the judged trajectory must be the official ones."""

    def test_de_impl_instruction_is_the_official_prompt(self):
        prompt = official_de_task_prompt(
            official_root=OFFICIAL_DACOMP_ROOT,
            task_type="impl",
            source_dir=OFFICIAL_DACOMP_ROOT / "dacomp-de" / "tasks" / "dacomp-de-impl-001",
        )
        # Sections the paraphrased instruction used to drop. The dbt ref() examples
        # matter most: emitting ref() is the standard failure mode on this track.
        for marker in (
            "You are a professional Data Engineer responsible for implementing",
            "## Project Background",
            "Use Pure DuckDB Syntax",
            "Do NOT** use the dbt `ref()` function",
            "**Correct ✅**",
            "**Incorrect ❌**",
            "## Contract file paths:",
            "- ./docs/data_contract.yaml",
            "## Data Contract Documentation (docs/data_contract.yaml):",
        ):
            self.assertIn(marker, prompt)

    def test_de_evol_instruction_leads_with_the_official_question_section(self):
        source_dir = OFFICIAL_DACOMP_ROOT / "dacomp-de" / "tasks" / "dacomp-de-evol-001"
        prompt = official_de_task_prompt(
            official_root=OFFICIAL_DACOMP_ROOT, task_type="evol", source_dir=source_dir
        )
        question = (source_dir / "question.md").read_text(encoding="utf-8")
        self.assertTrue(
            prompt.startswith("## Specific Business Requirements (from question.md):")
        )
        self.assertIn(question, prompt)
        self.assertLess(
            prompt.index("## Specific Business Requirements"),
            prompt.index("## Task Description:"),
            "the official evol prompt places question.md before the task description",
        )
        self.assertIn("modifying the `run.py` file is prohibited", prompt)

    def test_dataset_serves_the_official_prompt_as_the_task_query(self):
        task = DACompDataset(OFFICIAL_DACOMP_ROOT).load(
            DACompTrack.DE_IMPL, task_ids=["dacomp-de-impl-001"]
        )[0]
        expected = official_de_task_prompt(
            official_root=OFFICIAL_DACOMP_ROOT,
            task_type="impl",
            source_dir=task.source_dir,
        )
        self.assertEqual(task.instruction, expected)
        self.assertEqual(task.task_spec().query, expected)

    def test_da_trajectory_file_uses_the_official_renderer(self):
        trajectory = {
            "Task": "Analyze the portfolio.",
            "system_message": "system",
            "trajectory": [
                {
                    "thought": "Inspect the schema first.",
                    "action": "Bash(code='ls')",
                    "observation": "credit.sqlite",
                    "response": "raw model text",
                },
                {
                    "thought": "Compute the exposure.",
                    "action": "SQL(code='select 1')",
                    "observation": "1",
                    "response": "raw model text 2",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_dir = base / "dacomp-001"
            workspace_root = base / "workspace"
            run_dir.mkdir()
            workspace_root.mkdir()
            write_da_stage1_artifacts(
                run_dir=run_dir,
                workspace_root=workspace_root,
                official_root=OFFICIAL_DACOMP_ROOT,
                instance_id="dacomp-001",
                answer="report body",
                trajectory=trajectory,
                result_files={},
                finished=True,
                steps=2,
                error=None,
            )
            rendered = (run_dir / "dacomp-001-traj.txt").read_text(encoding="utf-8")

        # The rubric channel reads this text, so it must be the official step
        # rendering rather than a JSON dump of the trajectory object.
        self.assertTrue(rendered.startswith("--- Step 0 ---"))
        self.assertIn("thought: Inspect the schema first.", rendered)
        self.assertIn("action: Bash(code='ls')", rendered)
        self.assertIn("observation: credit.sqlite", rendered)
        self.assertIn("--- Step 1 ---", rendered)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(rendered)
        # Envelope keys of the official result.json must not leak into the render.
        self.assertNotIn("system_message", rendered)


if __name__ == "__main__":
    unittest.main()
