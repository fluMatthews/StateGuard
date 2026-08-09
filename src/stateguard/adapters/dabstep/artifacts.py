from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


_AGGREGATE_LOCK = threading.Lock()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_config(path: Path, config: dict[str, Any]) -> None:
    """Write a simple official-compatible YAML mapping without storing secrets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in config.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_task_artifacts(
    *,
    run_root: Path,
    task_dir: Path,
    answer_entry: dict[str, Any],
    trajectory: list[dict[str, Any]],
    error: str | None,
) -> dict[str, Any]:
    task_dir.mkdir(parents=True, exist_ok=True)
    payload = {**answer_entry, "trajectory": trajectory, "error": error}
    write_json(task_dir / "result.json", payload)
    write_json(task_dir / "trajectory.json", trajectory)
    (task_dir / "logs.txt").write_text(
        "\n\n".join(json.dumps(item, ensure_ascii=False, default=str) for item in trajectory),
        encoding="utf-8",
    )

    with _AGGREGATE_LOCK:
        answers_file = run_root / "answers.jsonl"
        existing: list[dict[str, Any]] = []
        if answers_file.is_file():
            existing = [
                json.loads(line)
                for line in answers_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        existing = [
            row
            for row in existing
            if str(row.get("task_id")) != str(answer_entry["task_id"])
        ]
        existing.append(answer_entry)
        answers_file.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in existing),
            encoding="utf-8",
        )
        with (run_root / "logs.txt").open("a", encoding="utf-8") as handle:
            handle.write(
                f"Task id: {answer_entry['task_id']}\tAnswer: "
                f"{answer_entry['agent_answer']}\n"
            )
    return payload
