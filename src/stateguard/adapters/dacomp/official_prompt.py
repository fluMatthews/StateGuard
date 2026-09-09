"""Reuse DAComp's own prompt and trajectory text builders verbatim.

The Worker must see exactly the official prompt, and the official DA rubric judge
must see exactly the official trajectory rendering. Both live inside the benchmark
repository, so they are taken from there rather than restated here.

``format_trajectory`` is imported directly: ``methods/da-agent/get_results.py``
only depends on the standard library.

``create_de_task_prompt`` cannot be imported the same way, because its module
``evaluation.benchmarks.dacomp.run_infer_de`` pulls in the whole OpenHands stack at
import time. Its source is therefore extracted from the official file and executed
on its own, so the produced prompt text stays byte-identical to the official one
without requiring the agent runtime just to read a task.
"""

from __future__ import annotations

import ast
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

DE_PROMPT_FUNCTION = "create_de_task_prompt"
# create_de_task_prompt only consults docs/*.yaml keys, the docs/data_contract.yaml
# body, and question.md. The official get_task_files walks the entire task tree,
# which reaches 465 MB for some evol tasks, so the scoped reader below supplies the
# same entries at a fraction of the cost.
PROMPT_RELEVANT_FILES = ("question.md",)
PROMPT_RELEVANT_DIRS = ("docs",)


def official_format_trajectory(trajectory: list[dict[str, Any]], official_root: Path) -> str:
    """Render a DA stage-1 trajectory exactly as the official exporter does."""
    module_root = (official_root / "methods" / "da-agent").resolve(strict=True)
    root_string = str(module_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    import get_results  # noqa: PLC0415 - official module, resolved via sys.path

    return get_results.format_trajectory(trajectory)


def official_de_task_prompt(*, official_root: Path, task_type: str, source_dir: Path) -> str:
    """Build the official DE prompt for one task without importing OpenHands."""
    if task_type not in {"impl", "evol"}:
        raise ValueError(f"unsupported DE task type: {task_type}")
    builder = _load_de_prompt_builder(official_root.resolve(strict=True))
    return builder({"task_type": task_type, "source_dir": str(source_dir)}, "en")


@lru_cache(maxsize=4)
def _load_de_prompt_builder(official_root: Path) -> Callable[[dict[str, str], str], str]:
    source_path = (
        official_root
        / "methods"
        / "de-agent"
        / "evaluation"
        / "benchmarks"
        / "dacomp"
        / "run_infer_de.py"
    ).resolve(strict=True)
    function_source = _extract_function_source(source_path, DE_PROMPT_FUNCTION)
    namespace: dict[str, Any] = {
        "get_task_files": _scoped_task_files,
        # Only the parameter annotation refers to pandas, so a stand-in keeps the
        # official signature intact without importing it.
        "pd": SimpleNamespace(Series=object),
    }
    exec(compile(function_source, str(source_path), "exec"), namespace)  # noqa: S102
    builder = namespace[DE_PROMPT_FUNCTION]
    if not callable(builder):
        raise RuntimeError(f"{DE_PROMPT_FUNCTION} was not defined by {source_path}")
    return builder


def _extract_function_source(source_path: Path, function_name: str) -> str:
    text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(text, node, padded=True) or ""
    raise LookupError(f"{function_name} not found in {source_path}")


def _scoped_task_files(task_dir: str) -> dict[str, str]:
    """Return the task files the official prompt builder actually reads."""
    root = Path(task_dir)
    files: dict[str, str] = {}
    if not root.exists():
        return files
    candidates: list[Path] = [root / name for name in PROMPT_RELEVANT_FILES]
    for directory in PROMPT_RELEVANT_DIRS:
        candidates.extend(sorted(path for path in (root / directory).rglob("*")))
    for path in candidates:
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        try:
            files[relative] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            files[relative] = "[BINARY FILE]"
    return files
