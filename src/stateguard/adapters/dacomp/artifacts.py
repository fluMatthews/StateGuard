from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .official_prompt import official_format_trajectory


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_da_stage1_artifacts(
    *,
    run_dir: Path,
    workspace_root: Path,
    official_root: Path,
    instance_id: str,
    answer: str,
    trajectory: dict[str, Any],
    result_files: dict[str, Any],
    finished: bool,
    steps: int,
    error: str | None,
) -> dict[str, Any]:
    report_path = workspace_root / "stage1.md"
    report = report_path.read_text(encoding="utf-8") if report_path.is_file() else answer
    (run_dir / f"{instance_id}.md").write_text(report or "", encoding="utf-8")
    # Keep the three-stage baseline's native Stage-1 output alongside the
    # flattened file consumed by the rubric bridge.
    (run_dir / "stage1.md").write_text(report or "", encoding="utf-8")
    # The official DA rubric channel scores this trajectory rendering (it falls back
    # to the report only when the trajectory is empty), so it must be produced by the
    # official exporter rather than dumped as JSON.
    (run_dir / f"{instance_id}-traj.txt").write_text(
        official_format_trajectory(trajectory.get("trajectory") or [], official_root),
        encoding="utf-8",
    )
    payload = {
        "finished": finished,
        "steps": steps,
        "result": answer,
        "result_files": result_files,
        **trajectory,
    }
    if error is not None:
        payload["error"] = error
    write_json(run_dir / "result.json", payload)
    write_json(workspace_root / "da_agent" / "result.json", payload)
    write_json(run_dir / "da_agent" / "result.json", payload)
    return payload


def write_de_artifacts(
    *,
    run_dir: Path,
    instance_id: str,
    instruction: str,
    trajectory: list[dict[str, Any]],
    run_success: bool,
    run_output: str,
    error: str | None,
) -> dict[str, Any]:
    sql_files = list((run_dir / "sql").rglob("*.sql")) if (run_dir / "sql").exists() else []
    tool_actions = {"execute_bash", "execute_ipython_cell", "browser"}
    summary = {
        "total_steps": len(trajectory),
        "tool_calls": sum(1 for item in trajectory if item.get("action") in tool_actions),
        "bash_calls": sum(1 for item in trajectory if item.get("action") == "execute_bash"),
        "ipython_calls": sum(
            1 for item in trajectory if item.get("action") == "execute_ipython_cell"
        ),
        "finish_calls": sum(1 for item in trajectory if item.get("action") == "finish"),
    }
    payload = {
        "instance_id": instance_id,
        "instruction": instruction,
        "sql_files_generated": bool(sql_files),
        "run_py_success": run_success,
        "run_py_output": run_output,
        "summary": summary,
        "trajectory": trajectory,
        "status": "error" if error else ("success" if run_success else "incomplete"),
        "error": error,
    }
    write_json(run_dir / "result.json", payload)
    write_json(
        run_dir / "workspace_summary.json",
        {
            "sql_files_check": "\n".join(
                str(path.relative_to(run_dir)) for path in sql_files
            ),
            "sql_staging_count": str(
                sum(1 for path in sql_files if "staging" in path.parts)
            ),
            "sql_intermediate_count": str(
                sum(1 for path in sql_files if "intermediate" in path.parts)
            ),
            "sql_marts_count": str(
                sum(1 for path in sql_files if "marts" in path.parts)
            ),
            "total_files_in_workspace": str(
                sum(1 for path in run_dir.rglob("*") if path.is_file())
            ),
            "workspace_path": str(run_dir),
        },
    )
    return payload
