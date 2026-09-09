from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from stateguard.core.models import Message


_WORKER_DATA_ROOT = re.compile(r"(?:/[A-Za-z0-9_.:@+~-]+)+/worker_data(?=/|\b)")
_REACT_STEP_TIMESTAMP = re.compile(
    r'("done"\s*:\s*(?:true|false)\s*,\s*"timestamp"\s*:\s*")[^"]+'
    r'("\s*,\s*"raw_model_output")'
)
# The executor tags each flush of a cell's stdout with a "[stdout]" banner, so
# one print block can arrive as one chunk in the parent run and as several in a
# replay of the very same code. The captured text is identical either way, and
# the banner is presentation rather than content, so fold it away before the
# digest instead of letting flush timing decide whether a prefix replays.
_STDOUT_CHUNK = re.compile(r"\n*\[stdout\][ \t]*")
# openpyxl emits this presentation-only warning while reading some benchmark
# workbooks. Jupyter may flush it before or after stdout across identical runs.
_OPENPYXL_EXTENSION_WARNING = re.compile(
    r"\n*\[stderr\][^\n]*openpyxl/worksheet/_reader\.py:\d+: "
    r"UserWarning: Unknown extension is not supported and will be removed"
    r"\n[ \t]*warn\(msg\)\n*"
)
# The DSGym executor writes transient cell filenames into syntax/indentation
# errors. They are random per run and should not affect replay equivalence.
_RUNTIME_CELL_FILENAME = re.compile(r"\b\d{6,12}\.py\b")

# A Worker's opening step almost always lists the staged data directory, and the
# staging copies files in whatever order the filesystem hands back, so the same
# probe prints the same names in a different order on every run. That ordering
# is an artifact of staging rather than anything the Worker computed, so it is
# folded away before the digest. Both rules below are deliberately narrow: they
# fire only on bare data filenames, which no answer in this corpus is made of,
# so an ordered result the Worker actually produced can never be sorted here.
_DATA_FILENAME = r"[A-Za-z0-9_.-]+\.(?:json|jsonl|txt|md|csv|sqlite|xlsx|xls|parquet)"
# ['alien.sqlite', 'analysis_rules.md', ...] printed by os.listdir
_FILENAME_LIST_LITERAL = re.compile(
    r"\[\s*'(?:%s)'(?:\s*,\s*'(?:%s)')+\s*\]" % (_DATA_FILENAME, _DATA_FILENAME)
)
# "  FILE: alien.sqlite 3063808" rows emitted by os.walk-style probes
_FILE_ROW = re.compile(r"^(\s*)FILE:\s+(%s)(\s.*)?$" % _DATA_FILENAME)
# "  alien.sqlite" rows emitted under a DIR:/ROOT: header with no other decoration
_BARE_FILE_ROW = re.compile(r"^\s+(%s)\s*$" % _DATA_FILENAME)



def canonical_message(value: Message | Mapping[str, Any]) -> dict[str, str]:
    """Keep only fields that a model provider receives and normalize run paths."""
    if isinstance(value, Message):
        role, content, name = value.role, value.content, value.name
    else:
        role = str(value.get("role", ""))
        content = str(value.get("content", ""))
        raw_name = value.get("name")
        name = None if raw_name is None else str(raw_name)
    row = {
        "role": role,
        "content": normalize_runtime_paths(_normalize_tagged_json(content)),
    }
    if name:
        row["name"] = name
    return row


def _normalize_tagged_json(content: str) -> str:
    """Canonicalize structured Manager blocks before hashing only."""
    stripped = content.strip()
    for opening, closing in (
        ("<manager_observation>", "</manager_observation>"),
        ("<tool_result>", "</tool_result>"),
    ):
        if not (stripped.startswith(opening) and stripped.endswith(closing)):
            continue
        inner = stripped[len(opening) : -len(closing)].strip()
        try:
            value = json.loads(inner)
        except json.JSONDecodeError:
            return content
        normalized = _normalize_json_value(value)
        payload = json.dumps(
            normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return f"{opening}\n{payload}\n{closing}"
    return content


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_runtime_paths(value)
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    return value


def _fold_stdout_chunk(match: "re.Match[str]") -> str:
    """Drop a leading banner outright; elsewhere leave the line break it carried."""
    return "" if match.start() == 0 else "\n"


def normalize_runtime_paths(text: str) -> str:
    """Make run-local paths, file listings, and stdout chunking replay-stable."""
    normalized = _normalize_run_roots(text)
    normalized = _REACT_STEP_TIMESTAMP.sub(r"\1<RUNTIME_TIMESTAMP>\2", normalized)
    normalized = _OPENPYXL_EXTENSION_WARNING.sub("\n", normalized)
    normalized = _RUNTIME_CELL_FILENAME.sub("<RUNTIME_CELL>.py", normalized)
    normalized = _STDOUT_CHUNK.sub(_fold_stdout_chunk, normalized)
    normalized = _sort_listing_blocks(normalized)
    normalized = _sort_path_list_blocks(normalized)
    normalized = _FILENAME_LIST_LITERAL.sub(_sort_filename_list_literal, normalized)
    normalized = _sort_row_runs(normalized, _FILE_ROW, 2)
    return _sort_row_runs(normalized, _BARE_FILE_ROW, 1)


def _normalize_run_roots(text: str) -> str:
    roots = sorted(
        {match.group(0).rsplit("/worker_data", 1)[0] for match in _WORKER_DATA_ROOT.finditer(text)},
        key=len,
        reverse=True,
    )
    normalized = text
    for root in roots:
        normalized = normalized.replace(root, "<RUN_ROOT>")
    return normalized.replace("<RUN_ROOT>/worker_data", "<WORKER_DATA_ROOT>")


def _sort_listing_blocks(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 3:
        return text
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        output.append(line)
        header = line.strip()
        if header in {"worker_data/", "[stdout] worker_data/"}:
            index += 1
            block: list[str] = []
            while index < len(lines) and lines[index].startswith("  "):
                block.append(lines[index])
                index += 1
            output.extend(sorted(block))
            continue
        index += 1
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(output) + suffix


def _sort_path_list_blocks(text: str) -> str:
    """Sort contiguous run-local path lists emitted by os.walk-style probes."""
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(("<RUN_ROOT>/", "<WORKER_DATA_ROOT>/")):
            block: list[str] = []
            while index < len(lines) and lines[index].startswith(("<RUN_ROOT>/", "<WORKER_DATA_ROOT>/")):
                block.append(lines[index])
                index += 1
            output.extend(sorted(block))
            continue
        output.append(line)
        index += 1
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(output) + suffix


def _sort_filename_list_literal(match: "re.Match[str]") -> str:
    """Order a printed list of staged data filenames so staging order cannot leak."""
    names = re.findall(r"'([^']*)'", match.group(0))
    return "[" + ", ".join(f"'{name}'" for name in sorted(names)) + "]"


def _sort_row_runs(text: str, pattern: "re.Pattern[str]", name_group: int) -> str:
    """Sort each contiguous run of directory-probe rows by the filename it names."""
    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        if pattern.match(lines[index]):
            block: list[str] = []
            while index < len(lines) and pattern.match(lines[index]):
                block.append(lines[index])
                index += 1
            output.extend(sorted(block, key=lambda row: pattern.match(row).group(name_group)))
            continue
        output.append(lines[index])
        index += 1
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(output) + suffix


def messages_digest(values: Iterable[Message | Mapping[str, Any]]) -> str:
    payload = [canonical_message(value) for value in values]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_tree_digest(root: Path) -> str:
    """Fingerprint runtime Python and prompt files for replay compatibility."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".txt"}:
            continue
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
