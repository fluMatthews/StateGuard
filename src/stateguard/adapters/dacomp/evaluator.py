from __future__ import annotations

import csv
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import write_json


def official_da_stage1_rubrics(
    *,
    official_root: Path,
    run_dir: Path,
    instance_id: str,
    rubrics_model: str,
    language: str = "en",
) -> dict[str, Any]:
    """Run only DA's official rubric channel and report Accuracy/Completeness.

    This intentionally never constructs or calls either GSB channel. It is an LLM
    judge and therefore must only be called after explicit authorization by the user.
    """
    suite = official_root / "dacomp-da" / "evaluation_suite"
    suite_string = str(suite.resolve(strict=True))
    if suite_string not in sys.path:
        sys.path.insert(0, suite_string)
    pipeline = importlib.import_module("core.pipeline")
    prompts = importlib.import_module("core.prompts")
    runners = importlib.import_module("core.runners")
    loader = importlib.import_module("core.tasks_loader")
    scoring = importlib.import_module("core.rubric_scoring")

    metadata_root = suite / ("src" if language == "en" else "src_zh")
    metadata = loader.load_task_metadata(metadata_root)
    if instance_id not in metadata:
        raise KeyError(f"official DA rubric metadata missing for {instance_id}")
    answer_path = run_dir / f"{instance_id}.md"
    trajectory_path = run_dir / f"{instance_id}-traj.txt"
    record = {
        "instance_id": instance_id,
        "trajectory": trajectory_path.read_text(encoding="utf-8"),
        "answer": answer_path.read_text(encoding="utf-8"),
        "answer_path": str(answer_path),
        "trajectory_path": str(trajectory_path),
        "answer_images": [],
    }
    with tempfile.TemporaryDirectory(prefix="dacomp-da-rubric-") as tmp:
        model_file = Path(tmp) / "stateguard.jsonl"
        model_file.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        tasks = pipeline.build_tasks(
            model_file,
            metadata,
            {},
            True,
            False,
            False,
            prompts.get_rubric_prompt(language),
            "",
            "",
        )
        runners.run_rubrics_tasks(tasks, rubrics_model, 1)
    if not tasks or not tasks[0].rubrics_result:
        raise RuntimeError("official DA rubric evaluator returned no valid result")
    raw = tasks[0].rubrics_result
    dimensions = scoring.parse_rubric_dimension_scores(raw)
    maxima = json.loads((metadata_root / instance_id / "metadata.json").read_text(encoding="utf-8"))
    result = {
        "instance_id": instance_id,
        "evaluation": "official_da_rubrics_stage1_only",
        "dimensions": {
            key: {
                "score": float(dimensions.get(key, 0.0)),
                "max_score": float(maxima[key]),
                "percentage": (
                    100.0 * float(dimensions.get(key, 0.0)) / float(maxima[key])
                    if float(maxima[key])
                    else 0.0
                ),
            }
            for key in ("Accuracy", "Completeness")
        },
        "rubrics_result": raw,
        "excluded": ["Conclusiveness", "Readability", "Analytical Depth", "Visualization"],
    }
    write_json(run_dir / "evaluation_da_stage1_accuracy_completeness.json", result)
    return result


def official_de_cs_cfs(
    *,
    official_root: Path,
    prediction_root: Path,
    instance_id: str,
    output_dir: Path,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Invoke the official deterministic DE evaluator once for CS and once for CFS."""
    suite = official_root / "dacomp-de" / "evaluation_suite"
    evaluator = suite / "evaluate.py"
    config = suite / "evaluation_config_compare.yaml"
    gold = suite / "gold"
    outputs: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("cs", "cfs"):
        destination = output_dir / f"{instance_id}_{mode}.json"
        command = [
            python_executable,
            str(evaluator),
            "single",
            "--config",
            str(config),
            "--gold_dir",
            str(gold),
            "--pred_dir",
            str(prediction_root),
            "--example_id",
            instance_id,
            "--output",
            str(destination),
            "--force-rebuild",
            "--mode",
            mode,
        ]
        completed = subprocess.run(
            command,
            cwd=suite,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"official DE {mode} evaluator failed: {completed.stderr or completed.stdout}"
            )
        outputs[mode] = json.loads(destination.read_text(encoding="utf-8"))
    summary = {
        "instance_id": instance_id,
        "CS": outputs["cs"].get("final_score", 0.0),
        "CFS": outputs["cfs"].get("final_score", 0.0),
        "results": outputs,
    }
    write_json(output_dir / f"{instance_id}_cs_cfs_summary.json", summary)
    return summary
