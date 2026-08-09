from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from stateguard.core.models import TaskSpec


class DACompTrack(str, Enum):
    DA_STAGE1 = "da-stage1"
    DE_IMPL = "de-impl"
    DE_EVOL = "de-evol"


@dataclass(frozen=True)
class DACompTask:
    instance_id: str
    track: DACompTrack
    instruction: str
    source_dir: Path
    metadata: dict[str, Any]

    def task_spec(self) -> TaskSpec:
        # The source directory is exposed through the native benchmark workspace,
        # not copied by the generic TaskSpec staging hook.
        return TaskSpec(
            id=self.instance_id,
            query=self.instruction,
            guidelines=tuple(self.metadata.get("guidelines", ())),
            metadata={
                "benchmark": "dacomp",
                "track": self.track.value,
                "source_dir": str(self.source_dir),
            },
        )


class DACompDataset:
    """Load only public task inputs; gold/rubrics remain evaluator-only."""

    def __init__(self, dacomp_root: Path) -> None:
        self.root = dacomp_root.expanduser().resolve(strict=True)
        self.da_root = self.root / "dacomp-da"
        self.de_root = self.root / "dacomp-de"

    def load(
        self,
        track: DACompTrack | str,
        *,
        task_ids: Iterable[str] | None = None,
        start_index: int = 0,
        task_limit: int | None = None,
    ) -> tuple[DACompTask, ...]:
        selected_track = DACompTrack(track)
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if task_limit is not None and task_limit < 1:
            raise ValueError("task_limit must be positive")
        if selected_track is DACompTrack.DA_STAGE1:
            tasks = self._load_da()
        else:
            tasks = self._load_de(selected_track)
        wanted = set(task_ids or ())
        if wanted:
            found = {task.instance_id for task in tasks}
            missing = sorted(wanted.difference(found))
            if missing:
                raise KeyError(f"unknown DAComp task ids: {missing}")
            tasks = [task for task in tasks if task.instance_id in wanted]
        tasks = tasks[start_index:]
        if task_limit is not None:
            tasks = tasks[:task_limit]
        return tuple(tasks)

    def _load_da(self) -> list[DACompTask]:
        task_file = self.da_root / "tasks" / "dacomp-da.jsonl"
        if not task_file.is_file():
            raise FileNotFoundError(f"official DA task file not found: {task_file}")
        tasks: list[DACompTask] = []
        for line in task_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = str(row["instance_id"])
            source_dir = self.da_root / "tasks" / instance_id
            if not source_dir.is_dir():
                raise FileNotFoundError(f"DA task directory not found: {source_dir}")
            tasks.append(
                DACompTask(
                    instance_id=instance_id,
                    track=DACompTrack.DA_STAGE1,
                    instruction=str(row["instruction"]),
                    source_dir=source_dir,
                    metadata={"official_record": row},
                )
            )
        return tasks

    def _load_de(self, track: DACompTrack) -> list[DACompTask]:
        task_root = self.de_root / "tasks"
        prefix = "dacomp-de-impl-" if track is DACompTrack.DE_IMPL else "dacomp-de-evol-"
        tasks: list[DACompTask] = []
        for source_dir in sorted(task_root.glob(f"{prefix}*")):
            if not source_dir.is_dir():
                continue
            required = [source_dir / "config" / "layer_dependencies.yaml", source_dir / "run.py"]
            required.append(
                source_dir / "docs" / "data_contract.yaml"
                if track is DACompTrack.DE_IMPL
                else source_dir / "question.md"
            )
            if not all(path.is_file() for path in required):
                continue
            instruction = _de_public_instruction(track, source_dir)
            tasks.append(
                DACompTask(
                    instance_id=source_dir.name,
                    track=track,
                    instruction=instruction,
                    source_dir=source_dir,
                    metadata={"required_files": [str(path) for path in required]},
                )
            )
        return tasks


def _de_public_instruction(track: DACompTrack, source_dir: Path) -> str:
    if track is DACompTrack.DE_EVOL:
        question = (source_dir / "question.md").read_text(encoding="utf-8")
        return (
            "You are a professional Data Engineer. Modify and optimize the existing SQL "
            "pipeline to satisfy question.md. Inspect the existing schema and dependencies "
            "before editing, do not modify run.py, and ensure python run.py succeeds.\n\n"
            "## Specific Business Requirements (from question.md)\n\n"
            + question
        )
    contract = (source_dir / "docs" / "data_contract.yaml").read_text(encoding="utf-8")
    return (
        "You are a professional Data Engineer. Populate ./sql according to "
        "docs/data_contract.yaml and config/layer_dependencies.yaml. Use DuckDB SQL, "
        "reference upstream tables as schema.table, do not use dbt ref(), do not end SQL "
        "files with semicolons, and ensure python run.py succeeds.\n\n"
        "## Data Contract Documentation (docs/data_contract.yaml)\n\n```yaml\n"
        + contract
        + "\n```"
    )
