from __future__ import annotations

import importlib
import sys
from pathlib import Path


def official_question_score(
    *, official_runner_root: Path, agent_answer: str, reference_answer: str
) -> bool:
    """Call the downloaded official deterministic DABstep scorer."""
    root = official_runner_root.expanduser().resolve(strict=True)
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    module = importlib.import_module("dabstep_benchmark.evaluation.scorer")
    origin = Path(str(getattr(module, "__file__", ""))).resolve()
    if root not in origin.parents:
        raise ImportError(f"loaded DABstep scorer from unexpected location: {origin}")
    return bool(module.question_scorer(str(agent_answer), str(reference_answer)))
