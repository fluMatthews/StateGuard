from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stateguard.core.models import TaskSpec
from stateguard.harness.blind_view import assert_blind


@dataclass(frozen=True)
class LongDSPublicTurn:
    """The complete Manager-visible part of one LongDS turn."""

    task_key: str
    domain: str
    dataset_name: str
    raw_task_id: str
    turn_id: int
    context: str
    question: str
    data_root: Path
    extra_info: dict[str, Any]

    @property
    def unit_id(self) -> str:
        return f"{self.task_key}/turn_{self.turn_id}"

    def task_spec(self) -> TaskSpec:
        metadata = {
            "benchmark": "longds",
            "domain": self.domain,
            "dataset_name": self.dataset_name,
            "raw_task_id": self.raw_task_id,
            "turn_id": self.turn_id,
            "data_root": str(self.data_root),
        }
        assert_blind(metadata)
        return TaskSpec(
            id=self.unit_id,
            query=self.question,
            context=self.context,
            metadata=metadata,
        )


@dataclass(frozen=True)
class LongDSPrivateTurn:
    """Evaluator-only fields. Never attach this object to a StateGuard runtime."""

    turn_id: int
    ground_truth: Any


@dataclass(frozen=True)
class LongDSTurn:
    public: LongDSPublicTurn
    private: LongDSPrivateTurn


@dataclass(frozen=True)
class LongDSTask:
    task_key: str
    domain: str
    dataset_name: str
    raw_task_id: str
    turns: tuple[LongDSTurn, ...]

    @property
    def data_root(self) -> Path:
        if not self.turns:
            raise ValueError("LongDS task has no turns")
        return self.turns[0].public.data_root


class LongDSDataset:
    """Load the exact task_list.json -> task.json hierarchy used by longds.py."""

    def __init__(self, task_root: Path, data_root: Path | None = None) -> None:
        self.task_root = task_root.expanduser().resolve()
        # Match scripts/longds.py: Path(dataset_path).parents[1] / data / longds.
        self.data_root = (
            data_root.expanduser().resolve()
            if data_root is not None
            else self.task_root.parents[1] / "data" / "longds"
        )

    def load(
        self,
        *,
        start_index: int = 0,
        task_limit: int | None = None,
        turn_limit: int | None = None,
    ) -> tuple[LongDSTask, ...]:
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if task_limit is not None and task_limit < 0:
            raise ValueError("task_limit must be non-negative")
        if turn_limit is not None and turn_limit < 0:
            raise ValueError("turn_limit must be non-negative")

        task_list_path = self.task_root / "task_list.json"
        task_entries = json.loads(task_list_path.read_text(encoding="utf-8"))
        selected = task_entries[start_index:]
        if task_limit is not None:
            selected = selected[:task_limit]

        tasks: list[LongDSTask] = []
        for entry in selected:
            domain = str(entry["task_domain"])
            dataset_name = str(entry["dataset_name"])
            raw_task_id = str(entry["task_id"])
            task_key = f"{domain}/{dataset_name}/{raw_task_id}"
            task_path = self.task_root / domain / dataset_name / raw_task_id / "task.json"
            raw_turns = json.loads(task_path.read_text(encoding="utf-8"))
            if turn_limit is not None:
                raw_turns = raw_turns[:turn_limit]
            turn_data_root = self.data_root / domain / dataset_name / raw_task_id / "data"
            turns: list[LongDSTurn] = []
            for index, raw in enumerate(raw_turns, 1):
                turn_id = int(raw.get("turn_id", index))
                extra_info = dict(raw.get("extra_info") or {})
                # Defensive removal protects against non-standard task files.
                for forbidden in (
                    "answer",
                    "ground_truth",
                    "gold",
                    "gold_answer",
                    "reference_answer",
                    "reward_spec",
                    "judge",
                ):
                    extra_info.pop(forbidden, None)
                public = LongDSPublicTurn(
                    task_key=task_key,
                    domain=domain,
                    dataset_name=dataset_name,
                    raw_task_id=raw_task_id,
                    turn_id=turn_id,
                    context=str(raw.get("context", "")),
                    question=str(raw["question"]),
                    data_root=turn_data_root,
                    extra_info=extra_info,
                )
                private = LongDSPrivateTurn(
                    turn_id=turn_id,
                    ground_truth=raw.get("answer", ""),
                )
                turns.append(LongDSTurn(public, private))
            tasks.append(
                LongDSTask(task_key, domain, dataset_name, raw_task_id, tuple(turns))
            )
        return tuple(tasks)
