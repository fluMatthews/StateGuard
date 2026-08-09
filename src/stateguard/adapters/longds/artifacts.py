from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def write_official_artifacts(
    *,
    run_dir: Path,
    trajectory: dict[str, Any],
    turn_results: list[dict[str, Any]],
) -> None:
    """Write the same primary files as scripts/longds.py."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "traj.json", trajectory)
    _write_json(run_dir / "results.json", turn_results)

    code_lines: list[str] = []
    for turn in turn_results:
        code_lines.append(f"############## turn {turn['turn_id']}")
        for message in turn["trajectory"]:
            if message.get("role") != "assistant":
                continue
            for match in re.finditer(
                r"<python>(.*?)</python>", message.get("content", ""), re.DOTALL
            ):
                code_lines.extend((match.group(1).strip(), ""))
    (run_dir / "code.py").write_text("\n".join(code_lines), encoding="utf-8")


def write_eval(run_dir: Path, evaluated: list[dict[str, Any]]) -> None:
    _write_json(run_dir / "results_eval.json", evaluated)


def write_turn_backup(run_dir: Path, turn_number: int, conversation: list[dict]) -> None:
    backup = run_dir / "bak"
    backup.mkdir(parents=True, exist_ok=True)
    _write_json(backup / f"turn_{turn_number}_result.json", conversation)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
