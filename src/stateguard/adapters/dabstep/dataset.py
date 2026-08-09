from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from stateguard.core.models import TaskSpec


CONTEXT_FILENAMES = (
    "acquirer_countries.csv",
    "payments.csv",
    "merchant_category_codes.csv",
    "fees.json",
    "merchant_data.json",
    "manual.md",
    "payments-readme.md",
)


@dataclass(frozen=True)
class DABstepTask:
    task_id: str
    question: str
    guidelines: str
    level: str
    split: str
    context_dir: Path
    # Evaluator-only. It is deliberately absent from TaskSpec and every Manager view.
    reference_answer: str | None = None

    def task_spec(self) -> TaskSpec:
        return TaskSpec(
            id=self.task_id,
            query=self.question,
            guidelines=(self.guidelines,),
            metadata={
                "benchmark": "dabstep",
                "split": self.split,
                "level": self.level,
            },
            data_files=tuple(
                str((self.context_dir / name).resolve(strict=True))
                for name in CONTEXT_FILENAMES
            ),
        )


class DABstepDataset:
    """Load the locally downloaded official inputs without exposing dev gold."""

    def __init__(self, dabstep_root: Path) -> None:
        self.root = dabstep_root.expanduser().resolve(strict=True)
        self.context_dir = self.root / "data" / "context"
        self.tasks_dir = self.root / "data" / "tasks"
        missing = [
            str(self.context_dir / name)
            for name in CONTEXT_FILENAMES
            if not (self.context_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"DABstep context is incomplete: {missing}")

    def load(
        self,
        split: str = "default",
        *,
        task_ids: Iterable[str | int] | None = None,
        start_index: int = 0,
        task_limit: int | None = None,
    ) -> tuple[DABstepTask, ...]:
        if split not in {"default", "dev"}:
            raise ValueError("DABstep split must be 'default' or 'dev'")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if task_limit is not None and task_limit < 1:
            raise ValueError("task_limit must be positive")
        source = self.tasks_dir / ("dev.jsonl" if split == "dev" else "all.jsonl")
        if not source.is_file():
            raise FileNotFoundError(f"official DABstep task file not found: {source}")

        tasks: list[DABstepTask] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tasks.append(
                DABstepTask(
                    task_id=str(row["task_id"]),
                    question=str(row["question"]),
                    guidelines=str(row["guidelines"]),
                    level=str(row["level"]),
                    split=split,
                    context_dir=self.context_dir,
                    reference_answer=(str(row["answer"]) if split == "dev" else None),
                )
            )

        wanted = {str(value) for value in (task_ids or ())}
        if wanted:
            found = {task.task_id for task in tasks}
            missing_ids = sorted(wanted.difference(found))
            if missing_ids:
                raise KeyError(f"unknown DABstep task ids: {missing_ids}")
            tasks = [task for task in tasks if task.task_id in wanted]
        tasks = tasks[start_index:]
        if task_limit is not None:
            tasks = tasks[:task_limit]
        return tuple(tasks)
