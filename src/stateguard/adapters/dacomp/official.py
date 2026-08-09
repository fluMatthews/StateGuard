from __future__ import annotations

import importlib
import sys
from pathlib import Path

from stateguard.core.models import TaskSpec

from .dataset import DACompTask, DACompTrack


def official_de_task_spec(
    *, official_root: Path, task: DACompTask, language: str
) -> TaskSpec:
    """Build the Worker/Manager-visible query with the vendored official builder.

    Importing is deliberately lazy: DA-stage1 and injected unit-test workers do not
    need the heavyweight OpenHands environment. Evaluation-only files are never
    passed to the builder.
    """
    if task.track not in {DACompTrack.DE_IMPL, DACompTrack.DE_EVOL}:
        return task.task_spec()
    method_root = official_root / "methods" / "de-agent"
    root_string = str(method_root.resolve(strict=True))
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    runner = importlib.import_module("evaluation.benchmarks.dacomp.run_infer_de")
    pandas = importlib.import_module("pandas")
    task_type = "impl" if task.track is DACompTrack.DE_IMPL else "evol"
    instance = pandas.Series(
        {
            "id": task.instance_id,
            "instance_id": task.instance_id,
            "task_type": task_type,
            "source_dir": str(task.source_dir),
        }
    )
    instruction = runner.create_de_task_prompt(instance, language)
    instruction += runner._guidelines_block(language)
    instruction += "\n\n" + runner.AGENT_SUFFIX_BY_LANG.get(
        language, runner.AGENT_SUFFIX_BY_LANG["zh"]
    )
    base = task.task_spec()
    return TaskSpec(
        id=base.id,
        query=instruction,
        context=base.context,
        guidelines=base.guidelines,
        data_files=base.data_files,
        metadata={**base.metadata, "official_prompt_builder": "create_de_task_prompt"},
    )
