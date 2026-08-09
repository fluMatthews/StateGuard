from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def official_llm_judge(
    task_trajectories: list[dict[str, Any]],
    *,
    dsgym_root: Path,
    api_key: str | None = None,
    base_url: str | None = None,
    judge_model: str = "deepseek-v4-pro",
    max_workers: int = 15,
) -> list[dict[str, Any]]:
    """Call DSGym scripts/longds.py's evaluator rather than maintaining a fork."""
    root = dsgym_root.expanduser().resolve()
    script_path = root / "scripts" / "longds.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"official LongDS runner not found: {script_path}")
    for path in (str(root / "scripts"), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(
        "stateguard_longds_official_runner", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official LongDS runner: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.llm_judge_evaluate(
        task_trajectories,
        api_key=api_key,
        base_url=base_url,
        judge_model=judge_model,
        max_workers=max_workers,
    )

