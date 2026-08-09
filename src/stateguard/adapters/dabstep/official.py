from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .dataset import DABstepTask


@dataclass(frozen=True)
class OfficialBaselineModules:
    utils: ModuleType
    prompts: ModuleType


def load_official_baseline(official_runner_root: Path) -> OfficialBaselineModules:
    """Import the downloaded baseline rather than duplicating its model setup."""
    baseline = official_runner_root.expanduser().resolve(strict=True) / "baseline"
    required = (baseline / "utils.py", baseline / "prompts.py")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"DABstep official baseline is incomplete: {baseline}")
    baseline_string = str(baseline)
    if baseline_string not in sys.path:
        sys.path.insert(0, baseline_string)
    utils = _official_import("utils", baseline)
    prompts = _official_import("prompts", baseline)
    return OfficialBaselineModules(utils=utils, prompts=prompts)


def create_official_agent(
    *,
    modules: OfficialBaselineModules,
    model_id: str,
    api_base: str | None,
    api_key: str | None,
    max_steps: int,
    context_dir: Path,
) -> Any:
    if modules.utils.is_reasoning_llm(model_id):
        return modules.utils.create_code_agent_with_reasoning_llm(
            model_id,
            api_base,
            api_key,
            max_steps,
            str(context_dir),
        )
    return modules.utils.create_code_agent_with_chat_llm(
        model_id,
        api_base,
        api_key,
        max_steps,
    )


def official_task_prompt(
    *,
    task: DABstepTask,
    model_id: str,
    native_agent: Any,
    modules: OfficialBaselineModules,
) -> str:
    if modules.utils.is_reasoning_llm(model_id):
        task_prompt = modules.prompts.reasoning_llm_task_prompt.format(
            question=task.question,
            guidelines=task.guidelines,
        )
        prompt = native_agent.system_prompt + "\n" + task_prompt
        # This is the exact special handling in official baseline/run.py.
        native_agent.system_prompt = ""
        return prompt
    return modules.prompts.chat_llm_task_prompt.format(
        ctx_path=str(task.context_dir),
        question=task.question,
        guidelines=task.guidelines,
    )


def _official_import(name: str, expected_parent: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        origin = Path(str(getattr(existing, "__file__", ""))).resolve()
        if origin.parent != expected_parent:
            raise ImportError(
                f"top-level module {name!r} is already loaded from {origin}; "
                f"cannot safely load DABstep module from {expected_parent}"
            )
        return existing
    module = importlib.import_module(name)
    origin = Path(str(getattr(module, "__file__", ""))).resolve()
    if origin.parent != expected_parent:
        raise ImportError(f"loaded {name!r} from unexpected location: {origin}")
    return module
