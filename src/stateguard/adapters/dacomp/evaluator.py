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
    """Run only DA's official rubric channel and report its three dimensions.

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
                # Same normalization as the official aggregation in
                # core/rubric_scoring.py: (raw / metadata max) * 100, clipped to
                # [0, 100] so an over-generous judge cannot exceed the dimension.
                "percentage": (
                    min(
                        100.0,
                        max(0.0, 100.0 * float(dimensions.get(key, 0.0)) / float(maxima[key])),
                    )
                    if float(maxima[key]) > 0
                    else None
                ),
            }
            # "Conclusiveness" is the code-internal name for the paper's
            # Insightfulness dimension: core/rubric_scoring.normalize_dimension_name
            # folds "insight", "conclus" and the Chinese "结论" onto this label.
            for key in ("Accuracy", "Completeness", "Conclusiveness")
        },
        "rubrics_result": raw,
        "excluded": ["Readability", "Analytical Depth", "Visualization"],
    }
    write_json(run_dir / "evaluation_da_stage1_accuracy_completeness.json", result)
    return result


def official_da_stage1_gsb_depth(
    *,
    official_root: Path,
    run_dir: Path,
    instance_id: str,
    gsb_model: str,
    language: str = "en",
) -> dict[str, Any]:
    """Run only DA's GSB text channel and report Analytical Depth.

    The visualization channel is never constructed: it needs the stage2 charts a
    stage1-only run does not produce, and the official text channel is compared
    without images anyway (core/pipeline.py passes include_images=False), so the
    absence of charts cannot reach this score. Readability comes back in the same
    response and is recorded per reference but is not reported as a dimension.

    This is an LLM judge and must only be called after explicit user authorization.
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
        raise KeyError(f"official DA GSB metadata missing for {instance_id}")
    answer_path = run_dir / f"{instance_id}.md"
    record = {
        "instance_id": instance_id,
        "trajectory": "",
        "answer": answer_path.read_text(encoding="utf-8"),
        "answer_path": str(answer_path),
        "answer_images": [],
    }
    with tempfile.TemporaryDirectory(prefix="dacomp-da-gsb-") as tmp:
        model_file = Path(tmp) / "stateguard.jsonl"
        model_file.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        tasks = pipeline.build_tasks(
            model_file,
            metadata,
            {},
            False,
            True,
            False,
            "",
            prompts.get_gsb_prompt(language, "text"),
            "",
        )
        if not tasks:
            raise RuntimeError("official DA GSB evaluator built no task")
        runners.run_gsb_channel(
            tasks,
            gsb_model,
            include_images=False,
            channel="text",
            max_workers=1,
        )
    task = tasks[0]
    per_reference: dict[str, Any] = {}
    for reference, content in (task.gsb_text_results or {}).items():
        scores = scoring.extract_gsb_scores(content or "")
        per_reference[reference] = {
            "analytical_depth_raw": scores.get("professionalism"),
            "readability_raw": scores.get("readability"),
        }
    depth_raw = [item["analytical_depth_raw"] for item in per_reference.values()]
    if not any(value is not None for value in depth_raw):
        raise RuntimeError("official DA GSB text channel returned no usable score")
    # Official aggregation: each reference is mapped to Good/Same/Bad at |raw|>3,
    # then max(0, (|G|-|B|)/(|G|+|S|+|B|)) -- core/rubric_scoring.trans_gsb_score.
    aggregate = scoring.trans_gsb_score(depth_raw)
    result = {
        "instance_id": instance_id,
        "evaluation": "official_da_gsb_text_stage1_only",
        "dimensions": {
            "Analytical Depth": {
                "score": aggregate,
                "percentage": None if aggregate is None else 100.0 * float(aggregate),
            }
        },
        "per_reference": per_reference,
        "gsb_text_results": dict(task.gsb_text_results or {}),
        "excluded": ["Visualization"],
    }
    write_json(run_dir / "evaluation_da_stage1_gsb_depth.json", result)
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
