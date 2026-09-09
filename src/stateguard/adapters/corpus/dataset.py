from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol

from stateguard.core.models import TaskSpec
from stateguard.harness.blind_view import assert_blind


CorpusMode = Literal["multi_turn", "single_query"]


@dataclass(frozen=True)
class CorpusPublicUnit:
    source: str
    raw_task_id: str
    mode: CorpusMode
    unit_index: int
    query: str
    context: str
    data_root: Path
    data_files: tuple[Path, ...]
    guidelines: tuple[str, ...] = ()

    @property
    def unit_id(self) -> str:
        if self.mode == "multi_turn":
            return f"{self.source}/{self.raw_task_id}/turn_{self.unit_index}"
        return f"{self.source}/{self.raw_task_id}"

    def task_spec(self) -> TaskSpec:
        metadata = {
            "benchmark": "corpus",
            "source_corpus": self.source,
            "raw_task_id": self.raw_task_id,
            "mode": self.mode,
            "unit_index": self.unit_index,
            "data_root": str(self.data_root),
        }
        assert_blind(metadata)
        return TaskSpec(
            id=self.unit_id,
            query=self.query,
            context=self.context,
            guidelines=self.guidelines,
            metadata=metadata,
            data_files=tuple(str(path) for path in self.data_files),
        )


@dataclass(frozen=True)
class CorpusPrivateUnit:
    reference_answer: Any
    expected_relation: str | None = None
    expected_upstream: tuple[int, ...] = ()
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class CorpusUnit:
    public: CorpusPublicUnit
    private: CorpusPrivateUnit


@dataclass(frozen=True)
class CorpusTask:
    source: str
    raw_task_id: str
    mode: CorpusMode
    units: tuple[CorpusUnit, ...]

    @property
    def task_key(self) -> str:
        return f"{self.source}/{self.mode}/{self.raw_task_id}"

    @property
    def data_root(self) -> Path:
        if not self.units:
            raise ValueError("corpus task has no units")
        return self.units[0].public.data_root


class CorpusLoader(Protocol):
    source: str

    def load(
        self,
        *,
        mode: CorpusMode | None = None,
        task_ids: Iterable[str] | None = None,
        start_index: int = 0,
        task_limit: int | None = None,
        unit_limit: int | None = None,
    ) -> tuple[CorpusTask, ...]: ...


class DSBenchV1Loader:
    source = "dsbench_v1"

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve(strict=True)

    def load(
        self,
        *,
        mode: CorpusMode | None = None,
        task_ids: Iterable[str] | None = None,
        start_index: int = 0,
        task_limit: int | None = None,
        unit_limit: int | None = None,
    ) -> tuple[CorpusTask, ...]:
        _validate_selection(start_index, task_limit, unit_limit)
        modes: tuple[CorpusMode, ...] = (
            (mode,) if mode is not None else ("multi_turn", "single_query")
        )
        tasks: list[CorpusTask] = []
        for selected_mode in modes:
            tasks.extend(self._load_mode(selected_mode, unit_limit))
        return _select_tasks(tasks, task_ids, start_index, task_limit)

    def _load_mode(
        self, mode: CorpusMode, unit_limit: int | None
    ) -> list[CorpusTask]:
        directory_name = "multi-turn" if mode == "multi_turn" else "single-turn"
        mode_root = self.root / directory_name
        answers_path = mode_root / "answers.json"
        if not answers_path.is_file():
            raise FileNotFoundError(f"missing DSBench-v1 answers: {answers_path}")
        answers = json.loads(answers_path.read_text(encoding="utf-8"))
        tasks: list[CorpusTask] = []
        for raw_task_id, answer_entry in answers.items():
            task_dir = mode_root / raw_task_id
            if not task_dir.is_dir():
                raise FileNotFoundError(f"missing DSBench-v1 task directory: {task_dir}")
            context = _read_required(task_dir / "introduction.txt")
            data_files = _task_input_files(task_dir)
            if mode == "multi_turn":
                question_ids = list(answer_entry.get("questions") or ())
                references = list(answer_entry.get("answers") or ())
                if len(question_ids) != len(references):
                    raise ValueError(
                        f"question/answer length mismatch for DSBench-v1 {raw_task_id}"
                    )
                units = [
                    CorpusUnit(
                        CorpusPublicUnit(
                            self.source,
                            raw_task_id,
                            mode,
                            index,
                            _read_required(task_dir / f"{question_id}.txt"),
                            context,
                            task_dir,
                            data_files,
                        ),
                        CorpusPrivateUnit(reference, provenance={"question_id": question_id}),
                    )
                    for index, (question_id, reference) in enumerate(
                        zip(question_ids, references), 1
                    )
                ]
            else:
                units = [
                    CorpusUnit(
                        CorpusPublicUnit(
                            self.source,
                            raw_task_id,
                            mode,
                            1,
                            _read_required(task_dir / "question.txt"),
                            context,
                            task_dir,
                            data_files,
                        ),
                        CorpusPrivateUnit(
                            answer_entry.get("answer"),
                            provenance={
                                "format": answer_entry.get("format"),
                                "pack": answer_entry.get("pack"),
                            },
                        ),
                    )
                ]
            if unit_limit is not None:
                units = units[:unit_limit]
            if units:
                tasks.append(CorpusTask(self.source, raw_task_id, mode, tuple(units)))
        return tasks


class IDABenchV2Loader:
    source = "idabench_v2"

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve(strict=True)

    def load(
        self,
        *,
        mode: CorpusMode | None = None,
        task_ids: Iterable[str] | None = None,
        start_index: int = 0,
        task_limit: int | None = None,
        unit_limit: int | None = None,
    ) -> tuple[CorpusTask, ...]:
        _validate_selection(start_index, task_limit, unit_limit)
        if mode not in {None, "multi_turn"}:
            raise ValueError("IDA-Bench-v2 corpus contains only multi-turn tasks")
        tasks: list[CorpusTask] = []
        for task_dir in sorted(self.root.iterdir(), key=lambda path: path.name):
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue
            task_path = task_dir / "task.json"
            answers_path = task_dir / "answers.json"
            data_root = task_dir / "data"
            if not task_path.is_file():
                continue
            if not answers_path.is_file() or not data_root.is_dir():
                raise FileNotFoundError(f"incomplete IDA-Bench-v2 task: {task_dir}")
            raw_turns = json.loads(task_path.read_text(encoding="utf-8"))
            answer_turns = json.loads(answers_path.read_text(encoding="utf-8"))["turns"]
            answers_by_id = {int(row["turn_id"]): row for row in answer_turns}
            data_files = tuple(
                path.resolve()
                for path in sorted(data_root.rglob("*"))
                if path.is_file()
            )
            units: list[CorpusUnit] = []
            for index, row in enumerate(raw_turns, 1):
                turn_id = int(row.get("turn_id", index))
                answer_row = answers_by_id.get(turn_id)
                if answer_row is None:
                    raise ValueError(f"missing answer for {task_dir.name} turn {turn_id}")
                units.append(
                    CorpusUnit(
                        CorpusPublicUnit(
                            self.source,
                            task_dir.name,
                            "multi_turn",
                            turn_id,
                            str(row["question"]),
                            "",
                            data_root,
                            data_files,
                        ),
                        CorpusPrivateUnit(
                            answer_row.get("answer"),
                            expected_relation=str(row.get("relation") or "") or None,
                            expected_upstream=tuple(int(value) for value in row.get("upstream", ())),
                            provenance={"source_shards": list(row.get("source_shards", ()))},
                        ),
                    )
                )
            if unit_limit is not None:
                units = units[:unit_limit]
            if units:
                tasks.append(
                    CorpusTask(self.source, task_dir.name, "multi_turn", tuple(units))
                )
        return _select_tasks(tasks, task_ids, start_index, task_limit)


class StateGuardV6Loader:
    """Read the StateGuard SFT corpus v6 release, which is single-query only.

    The release keeps the agent-visible bundle under ``tasks/single_query`` and
    every hidden label under ``labels``. A task states its objective in
    ``question.json`` and its output contract in that file's ``guidelines``, so
    both travel to the runtime instead of being flattened into the query. The
    published ``visible_files`` list is advisory and is not always complete, so
    the staged data is enumerated from ``files/`` the way the other loaders do.
    """

    source = "stateguard_v6"

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve(strict=True)

    def load(
        self,
        *,
        mode: CorpusMode | None = None,
        task_ids: Iterable[str] | None = None,
        start_index: int = 0,
        task_limit: int | None = None,
        unit_limit: int | None = None,
    ) -> tuple[CorpusTask, ...]:
        _validate_selection(start_index, task_limit, unit_limit)
        if mode not in {None, "single_query"}:
            raise ValueError("StateGuard corpus v6 contains only single-query tasks")
        tasks_root = self.root / "tasks" / "single_query"
        labels_root = self.root / "labels"
        if not tasks_root.is_dir():
            raise FileNotFoundError(f"missing StateGuard v6 tasks: {tasks_root}")
        tasks: list[CorpusTask] = []
        for task_dir in sorted(tasks_root.iterdir(), key=lambda path: path.name):
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue
            question = json.loads(
                _read_required(task_dir / "question.json")
            )
            if str(question.get("task_type", "single_query")) != "single_query":
                continue
            data_root = task_dir / "files"
            if not data_root.is_dir():
                raise FileNotFoundError(f"missing StateGuard v6 data: {data_root}")
            answers_path = labels_root / task_dir.name / "answers.json"
            if not answers_path.is_file():
                raise FileNotFoundError(f"missing StateGuard v6 answers: {answers_path}")
            answers = json.loads(answers_path.read_text(encoding="utf-8"))["answers"]
            if not answers:
                raise ValueError(f"StateGuard v6 task has no answer: {task_dir.name}")
            reference = (
                answers[0]["value"]
                if len(answers) == 1
                else {str(row["answer_id"]): row["value"] for row in answers}
            )
            data_files = tuple(
                path.resolve()
                for path in sorted(data_root.rglob("*"))
                if path.is_file()
            )
            unit = CorpusUnit(
                CorpusPublicUnit(
                    self.source,
                    task_dir.name,
                    "single_query",
                    1,
                    str(question["question"]),
                    _read_required(task_dir / "introduction.md"),
                    data_root,
                    data_files,
                    tuple(str(item) for item in question.get("guidelines") or ()),
                ),
                CorpusPrivateUnit(
                    reference,
                    provenance={
                        "level": question.get("level"),
                        "answer_ids": [str(row["answer_id"]) for row in answers],
                    },
                ),
            )
            if unit_limit is not None and unit_limit < 1:
                continue
            tasks.append(
                CorpusTask(self.source, task_dir.name, "single_query", (unit,))
            )
        return _select_tasks(tasks, task_ids, start_index, task_limit)


class StateGuardMediumSingleLoader:
    """Read the StateGuard SFT Single-Query Medium release, single-query only.

    Every task keeps its bundle under ``tasks/<task_id>`` and its labels under
    ``labels/<task_id>``. Metric definitions live in ``files/analysis_rules.md``
    beside the data, so the query stays short and the Worker reads the rules the
    way it reads any other supplied file. The published ``guidelines`` object is
    the answer contract rather than prose; it is rendered into the lines a model
    can act on so the Manager can check the shape of a submitted answer.

    Staged data comes from the task's declared ``files`` list rather than from a
    directory walk. SQLite leaves ``-wal`` and ``-shm`` sidecars next to a
    database whenever one is opened, so a walk would hand the Worker inputs the
    task never declared and would make the staged set depend on whether anything
    had recently read the database.
    """

    source = "stateguard_medium_single"

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve(strict=True)

    def load(
        self,
        *,
        mode: CorpusMode | None = None,
        task_ids: Iterable[str] | None = None,
        start_index: int = 0,
        task_limit: int | None = None,
        unit_limit: int | None = None,
    ) -> tuple[CorpusTask, ...]:
        _validate_selection(start_index, task_limit, unit_limit)
        if mode not in {None, "single_query"}:
            raise ValueError(
                "StateGuard medium single-query corpus contains only single-query tasks"
            )
        tasks_root = self.root / "tasks"
        labels_root = self.root / "labels"
        if not tasks_root.is_dir():
            raise FileNotFoundError(f"missing medium single-query tasks: {tasks_root}")
        tasks: list[CorpusTask] = []
        for task_dir in sorted(tasks_root.iterdir(), key=lambda path: path.name):
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue
            question = json.loads(_read_required(task_dir / "question.json"))
            if str(question.get("task_type", "single_query")) != "single_query":
                continue
            data_root = task_dir / "files"
            if not data_root.is_dir():
                raise FileNotFoundError(f"missing medium single-query data: {data_root}")
            answers_path = labels_root / task_dir.name / "answers.json"
            if not answers_path.is_file():
                raise FileNotFoundError(
                    f"missing medium single-query answers: {answers_path}"
                )
            answer = json.loads(answers_path.read_text(encoding="utf-8"))["answer"]
            data_files = _declared_data_files(
                task_dir, data_root, question.get("files")
            )
            contract = _render_answer_contract(question.get("guidelines"))
            unit = CorpusUnit(
                CorpusPublicUnit(
                    self.source,
                    task_dir.name,
                    "single_query",
                    1,
                    str(question["question"]),
                    _with_answer_contract(
                        _read_required(task_dir / "introduction.md"), contract
                    ),
                    data_root,
                    data_files,
                    contract,
                ),
                CorpusPrivateUnit(
                    answer["value"],
                    provenance={
                        "level": question.get("level"),
                        "title": question.get("title"),
                    },
                ),
            )
            if unit_limit is not None and unit_limit < 1:
                continue
            tasks.append(
                CorpusTask(self.source, task_dir.name, "single_query", (unit,))
            )
        return _select_tasks(tasks, task_ids, start_index, task_limit)


def _with_answer_contract(context: str, contract: tuple[str, ...]) -> str:
    """Fold the answer contract into the context the Worker actually receives.

    The official DSGym turn prompt is built from context and question alone, so
    a contract carried only in TaskSpec.guidelines reaches the Manager and never
    the Worker. This release states its required keys nowhere else, and a Worker
    that cannot see them answers correctly in prose and is then scored wrong for
    the shape rather than the analysis.
    """
    if not contract:
        return context
    return context.rstrip() + "\n\nAnswer contract:\n" + "\n".join(
        f"- {line}" for line in contract
    )


def _declared_data_files(
    task_dir: Path, data_root: Path, declared: Any
) -> tuple[Path, ...]:
    """Resolve a task's declared file list, refusing anything outside its data root."""
    if not declared:
        raise ValueError(f"task declares no input files: {task_dir.name}")
    resolved: list[Path] = []
    seen: set[Path] = set()
    for entry in declared:
        path = (task_dir / str(entry)).resolve()
        try:
            path.relative_to(data_root.resolve())
        except ValueError:
            raise ValueError(
                f"declared corpus file is outside its data root: {entry}"
            ) from None
        if not path.is_file():
            raise FileNotFoundError(f"declared corpus file is missing: {path}")
        if path in seen:
            raise ValueError(f"declared corpus file is listed twice: {entry}")
        seen.add(path)
        resolved.append(path)
    return tuple(sorted(resolved))


def _render_answer_contract(guidelines: Any) -> tuple[str, ...]:
    """Turn a structured answer contract into the lines a model can act on."""
    if not isinstance(guidelines, dict):
        return tuple(str(item) for item in guidelines or ())
    lines: list[str] = []
    fields = guidelines.get("fields") or []
    if fields:
        rendered = ", ".join(
            f"{row['name']} ({row['type']})" if row.get("type") else str(row["name"])
            for row in fields
        )
        form = guidelines.get("format") or "one JSON object"
        lines.append(f"Return {form} with exactly these keys: {rendered}.")
    if guidelines.get("extra_fields_allowed") is False:
        lines.append("Do not add any key beyond the ones listed.")
    tolerance = guidelines.get("numeric_tolerance")
    if tolerance:
        lines.append(f"Numeric answers are accepted within {tolerance}.")
    return tuple(lines)


def create_corpus_loader(source: str, root: Path) -> CorpusLoader:
    normalized = source.strip().lower().replace("-", "_")
    if normalized in {"dsbench", "dsbench_v1"}:
        return DSBenchV1Loader(root)
    if normalized in {"idabench", "idabench_v2", "ida_bench_v2"}:
        return IDABenchV2Loader(root)
    if normalized in {"stateguard_v6", "stateguard_sft_corpus_v6"}:
        return StateGuardV6Loader(root)
    if normalized in {"stateguard_medium_single", "stateguard_sq_medium_v2"}:
        return StateGuardMediumSingleLoader(root)
    raise ValueError(f"unsupported corpus source: {source}")


def _task_input_files(task_dir: Path) -> tuple[Path, ...]:
    ignored = {"introduction.txt", "question.txt"}
    files = [
        path.resolve()
        for path in sorted(task_dir.iterdir(), key=lambda item: item.name)
        if path.is_file()
        and path.name not in ignored
        and not (path.name.startswith("question") and path.suffix == ".txt")
        and path.name not in {"answers.json", "relation_graph.json"}
    ]
    return tuple(files)


def _read_required(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = path.read_text(encoding="utf-8", errors="replace").strip()
    if not value:
        raise ValueError(f"empty corpus text file: {path}")
    return value


def _select_tasks(
    tasks: list[CorpusTask],
    task_ids: Iterable[str] | None,
    start_index: int,
    task_limit: int | None,
) -> tuple[CorpusTask, ...]:
    tasks = sorted(tasks, key=lambda task: (task.mode, task.raw_task_id))
    wanted = {str(value) for value in (task_ids or ())}
    if wanted:
        found = {task.raw_task_id for task in tasks}
        missing = sorted(wanted.difference(found))
        if missing:
            raise KeyError(f"unknown corpus task ids: {missing}")
        tasks = [task for task in tasks if task.raw_task_id in wanted]
    tasks = tasks[start_index:]
    if task_limit is not None:
        tasks = tasks[:task_limit]
    return tuple(tasks)


def _validate_selection(
    start_index: int, task_limit: int | None, unit_limit: int | None
) -> None:
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if task_limit is not None and task_limit < 1:
        raise ValueError("task_limit must be positive")
    if unit_limit is not None and unit_limit < 1:
        raise ValueError("unit_limit must be positive")
