from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from stateguard.core.models import TaskSpec

from .official_prompt import official_de_task_prompt


class DACompTrack(str, Enum):
    DA_STAGE1 = "da-stage1"
    DE_IMPL = "de-impl"
    DE_EVOL = "de-evol"


# The statement handed to the Worker is also what the Manager reviews against,
# and DE-Impl's runs far past what a review needs: it embeds the whole
# data_contract.yaml, so the Manager's pinned prompt reached 165,716 characters
# -- 94% of a 40,960-token window, against 0.2% for the observation it is meant
# to reason about. That message is pinned, so no context budget could reach it
# and the window was full before the first review.
#
# Only this track has that shape. In all 30 DE-Impl tasks the contract opens at
# character 2,656, after the fixed task description, objectives and output
# standard, and runs to the last byte -- 94% to 99% of the text -- so the head
# carries every instruction and the elided tail is table definitions, which the
# Manager reads off the Worker's trajectory anyway. DE-Evol shares none of it:
# no contract block, 12,423 to 39,438 characters of requirements written for
# that task alone, at most 28% of the window. It fits, every byte of it is what
# the review is judged against, and it is passed through whole like every other
# benchmark's.
# What survives the cut above is the fixed preamble every DE-Impl task shares:
# a role description, four formatting rules and a correct/incorrect dbt example.
# It names no table, no grain and no rule, which is why the Manager's committed
# states listed "no semicolon at file end" as their constraints -- it was
# repeating the only spec it could see -- and why one run pushed the Worker to
# cast eight staging tables the contract never asked to be cast, costing 15
# points. The tail that used to follow is gone: it landed wherever the cut fell,
# which for one task was the last model's column descriptions.
#
# What replaces it is the contract's executable part. Three fields carry
# instructions and the rest describe:
#   - validation_rules, a column's rule plus on_failure (correct, delete_row or
#     nullify_field), present in 28 of the 30 tasks
#   - source_expression, the derivation written out, in 3 tasks
#   - row_filters, the same shape at table level, in 3 tasks
# Lineage (source_table or source_models) and grain are in all 30 and are what
# let a review reason about a dependency at all.
#
# A column appears only when it carries one of those; the rest pass through and
# would cost 22% of the contract's bytes to list. data_type stays out even
# though it is 19% of the bytes: gold does not enforce it where no expression is
# given, and reading it as a mandate is exactly the mistake that cost the 15
# points. description (28%) and business_logic (15%) stay out as prose -- and
# business_logic reaches the Manager anyway through the Worker's own reading of
# the contract, in the observation.
#
# The result runs 7,135 characters at the median and 20,186 at most, against
# 9,163 and 12,079 for the head/tail/lineage form it replaces.
_CONTRACT_BUDGET = 24_000

_C_TABLE = re.compile(r"^(\s*)-\s+name:\s*['\"]?([A-Za-z0-9_]+)['\"]?\s*$")
_C_FIELD = re.compile(
    r"^\s*(source_table|grain|source_expression|constraints|on_failure|data_type):\s*(.+)$"
)
_C_RULE = re.compile(r"^(\s*)-\s+rule:\s*(.+)$")
_C_ITEM = re.compile(r"^\s*-\s+['\"]?([A-Za-z0-9_.]+)['\"]?\s*$")


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _contract_digest(contract: str, budget: int = _CONTRACT_BUDGET) -> str:
    """The contract's lineage, grain and per-column instructions.

    Scanned as text rather than parsed: one contract (dacomp-de-impl-020) holds
    a regex whose backslash escape makes yaml.safe_load raise, and that task
    should still get a digest. An entry counts as a table only when a
    ``columns:`` key follows it, which separates tables from the column entries
    that share the ``- name:`` shape.
    """
    lines = contract.splitlines()
    tables: list[dict[str, Any]] = []
    table: dict[str, Any] | None = None
    column: dict[str, Any] | None = None
    for index, line in enumerate(lines):
        entry = _C_TABLE.match(line)
        if entry:
            name, indent = entry.group(2), len(entry.group(1))
            follows_columns = False
            cursor = index + 1
            while cursor < len(lines):
                ahead = lines[cursor]
                if ahead.strip() and (len(ahead) - len(ahead.lstrip())) <= indent:
                    break
                if ahead.lstrip().startswith("columns:"):
                    follows_columns = True
                    break
                cursor += 1
            if follows_columns:
                table = {"name": name, "sources": [], "grain": "", "columns": [], "filters": []}
                tables.append(table)
                column = None
            elif table is not None:
                column = {
                    "name": name,
                    "type": "",
                    "expression": "",
                    "rules": [],
                    "constraints": "",
                }
                table["columns"].append(column)
            continue
        if table is None:
            continue
        if line.lstrip().startswith("source_models:"):
            cursor = index + 1
            while cursor < len(lines):
                item = _C_ITEM.match(lines[cursor])
                if not item:
                    break
                table["sources"].append(item.group(1))
                cursor += 1
            continue
        rule = _C_RULE.match(line)
        if rule:
            # Only a wrapping quote is YAML's; a rule's own quotes are content,
            # as in "account_id ~ '^[A-Za-z0-9]{15,18}$'".
            text = _unquote(rule.group(2).strip())
            outcome = ""
            if index + 1 < len(lines):
                nxt = _C_FIELD.match(lines[index + 1])
                if nxt and nxt.group(1) == "on_failure":
                    outcome = nxt.group(2).strip()
            entry_text = f"{text} -> {outcome}" if outcome else text
            deep = len(rule.group(1)) >= 12
            if column is not None and deep:
                column["rules"].append(entry_text)
            else:
                table["filters"].append(entry_text)
            continue
        field = _C_FIELD.match(line)
        if not field:
            continue
        key, value = field.group(1), _unquote(field.group(2).strip())
        if key == "source_table":
            table["sources"] = [value]
        elif key == "grain":
            table["grain"] = value
        elif key == "source_expression" and column is not None:
            column["expression"] = value
        elif key == "constraints" and column is not None:
            column["constraints"] = value
        elif key == "data_type" and column is not None:
            column["type"] = value
    return _render_contract(tables, budget)


def _render_contract(tables: list[dict[str, Any]], budget: int) -> str:
    blocks: list[str] = []
    for table in tables:
        sources = ", ".join(table["sources"]) if table["sources"] else "(source unstated)"
        head = f"{table['name']} <- {sources}"
        if table["grain"]:
            head += f" | {table['grain']}"
        body = [head]
        # Grouped rather than one per line: 55% of columns declare a type other
        # than VARCHAR, and repeating the type costs 3,100 characters a task at
        # the median for nothing. VARCHAR is the default a passthrough already
        # has, so only the others are worth naming.
        by_type: dict[str, list[str]] = {}
        for column in table["columns"]:
            declared = column["type"]
            if declared and not declared.upper().startswith("VARCHAR"):
                by_type.setdefault(declared, []).append(column["name"])
        for declared, names in by_type.items():
            body.append(f"    {declared}: {', '.join(names)}")
        for column in table["columns"]:
            if column["expression"]:
                body.append(f"    {column['name']} = {column['expression']}")
            for rule in column["rules"]:
                body.append(f"    {column['name']}: {rule}")
            if not column["expression"] and not column["rules"] and column["constraints"]:
                body.append(f"    {column['name']} {column['constraints']}")
        for filter_text in table["filters"]:
            body.append(f"    row filter: {filter_text}")
        blocks.append("\n".join(body))
    if not blocks:
        return ""
    text = "\n".join(blocks)
    if len(text) <= budget:
        return text
    kept: list[str] = []
    for block in blocks:
        if sum(len(x) + 1 for x in kept) + len(block) > budget:
            break
        kept.append(block)
    dropped = len(blocks) - len(kept)
    return "\n".join(kept) + f"\n...[{dropped} more models omitted]..."


_IMPL_REVIEW_HEAD = 3_000


def _impl_review_statement(instruction: str) -> str:
    """The DE-Impl statement, minus the contract body the Manager cannot use."""
    marker = instruction.find("```yaml")
    if marker < 0 or len(instruction) <= _IMPL_REVIEW_HEAD:
        return instruction
    head = instruction[:_IMPL_REVIEW_HEAD]
    digest = _contract_digest(instruction[marker + 7 :])
    dropped = len(instruction) - _IMPL_REVIEW_HEAD
    if not digest:
        return f"{head}\n...[{dropped:,} characters of data contract elided]...\n"
    # No gloss on what the extract means. The previous version carried one --
    # that a column absent from the rules passes through unchanged -- and it
    # was wrong: across the 30 contracts gold honours the declared data_type
    # for 10,076 of the 10,386 columns that have no source_expression, 97.0%.
    # Asserting the opposite told the Manager to leave five TIMESTAMP columns
    # as text and cost 9.4 points on one task, the mirror of the 15 points the
    # run before lost by casting columns the contract never mentioned. What
    # follows is the contract's own content, reorganised, and nothing else.
    return (
        f"{head}\n...[{dropped:,} characters of data contract elided]...\n\n"
        f"{digest}\n"
    )


@dataclass(frozen=True)
class DACompTask:
    instance_id: str
    track: DACompTrack
    instruction: str
    source_dir: Path
    metadata: dict[str, Any]

    def data_file_names(self) -> tuple[str, ...]:
        """Relative paths the Manager's probe can open, exactly as it sees them.

        DACompProbeExecutor mirrors this same tree into every probe and keys
        data_files by these names. Without the list the Manager is initialized
        with "data_files": [] and concludes it has nothing to check: one run
        opened an in-memory database and wrote "No data files available to
        probe" without ever looking. Only the Manager reads this -- DE prepares
        its Worker with task.query alone, and initial_prompt() never renders
        data files -- so the Worker's input is unchanged.
        """
        return tuple(
            sorted(
                str(path.relative_to(self.source_dir))
                for path in self.source_dir.rglob("*")
                if path.is_file()
            )
        )

    def task_spec(self) -> TaskSpec:
        # The source directory is exposed through the native benchmark workspace,
        # not copied by the generic TaskSpec staging hook.
        metadata: dict[str, Any] = {
            "benchmark": "dacomp",
            "track": self.track.value,
            "source_dir": str(self.source_dir),
        }
        if self.track is DACompTrack.DE_IMPL:
            metadata["manager_query"] = _impl_review_statement(self.instruction)
        return TaskSpec(
            id=self.instance_id,
            query=self.instruction,
            guidelines=tuple(self.metadata.get("guidelines", ())),
            data_files=self.data_file_names(),
            metadata=metadata,
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
            instruction = official_de_task_prompt(
                official_root=self.root,
                task_type="impl" if track is DACompTrack.DE_IMPL else "evol",
                source_dir=source_dir,
            )
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
