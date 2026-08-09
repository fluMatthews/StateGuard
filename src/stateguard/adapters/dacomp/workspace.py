from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATABASE_SUFFIXES = {".duckdb", ".sqlite", ".sqlite3", ".db"}


@dataclass(frozen=True)
class DACompWorkspaceSnapshot:
    overrides: tuple[tuple[str, bytes], ...]
    deleted_source_files: tuple[str, ...]
    database_files: tuple[str, ...]
    modified_database_files: tuple[str, ...]
    fingerprint: tuple[tuple[str, int, int], ...]
    cleanup_events: tuple[dict[str, Any], ...]


class DACompWorkspace:
    """Task-local filesystem with source-relative, database-light checkpoints.

    DAComp inputs include multi-gigabyte SQLite/DuckDB files. Capturing those bytes at
    every state boundary is neither necessary nor viable. A snapshot stores only
    changed/new non-database files, deleted source files, database path/status metadata,
    and a cheap stat fingerprint. On a real rollback it restores the official source
    tree, reapplies text/code deltas, and reruns run.py when the checkpoint contained a
    generated/modified database. This treats databases as derived artifacts of the SQL
    project while keeping SQL/report changes exact.
    """

    checkpoint_mode = "source_relative_delta_and_derived_database_rebuild"

    def __init__(self, root: Path, source_dir: Path) -> None:
        self.root = root.expanduser().resolve()
        self.source_dir = source_dir.expanduser().resolve(strict=True)
        self.cleanup_events: list[dict[str, Any]] = []
        self._source_stats = _stats(self.source_dir)

    def prepare(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        shutil.copytree(self.source_dir, self.root, copy_function=_copy_file)

    def snapshot(self) -> DACompWorkspaceSnapshot:
        current_stats = _stats(self.root)
        overrides: list[tuple[str, bytes]] = []
        database_files: list[str] = []
        modified_databases: list[str] = []
        for relative, (size, mtime_ns) in current_stats.items():
            path = self.root / relative
            if path.suffix.lower() in DATABASE_SUFFIXES:
                database_files.append(relative)
                if self._source_stats.get(relative) != (size, mtime_ns):
                    modified_databases.append(relative)
                continue
            if self._source_stats.get(relative) != (size, mtime_ns):
                overrides.append((relative, path.read_bytes()))
        deleted = tuple(
            sorted(
                relative
                for relative in self._source_stats
                if relative not in current_stats
                and Path(relative).suffix.lower() not in DATABASE_SUFFIXES
            )
        )
        return DACompWorkspaceSnapshot(
            tuple(overrides),
            deleted,
            tuple(sorted(database_files)),
            tuple(sorted(modified_databases)),
            tuple((key, *value) for key, value in sorted(current_stats.items())),
            tuple(self.cleanup_events),
        )

    def restore(self, snapshot: DACompWorkspaceSnapshot) -> None:
        if _fingerprint(self.root) == snapshot.fingerprint:
            self.cleanup_events = list(snapshot.cleanup_events)
            return
        if self.root.exists():
            shutil.rmtree(self.root)
        shutil.copytree(self.source_dir, self.root, copy_function=_copy_file)
        for relative in snapshot.deleted_source_files:
            path = self.root / relative
            if path.is_file():
                path.unlink()
        for relative, content in snapshot.overrides:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        desired_databases = set(snapshot.database_files)
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix.lower() in DATABASE_SUFFIXES:
                relative = str(path.relative_to(self.root))
                if relative not in desired_databases:
                    path.unlink()
        if snapshot.modified_database_files:
            run_py = self.root / "run.py"
            if run_py.is_file():
                completed = subprocess.run(
                    [sys.executable, "run.py"],
                    cwd=self.root,
                    text=True,
                    capture_output=True,
                    timeout=300,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        "failed to rebuild DAComp derived database during rollback: "
                        + (completed.stderr or completed.stdout)
                    )
        self.cleanup_events = list(snapshot.cleanup_events)

    def manifest(self) -> dict[str, Any]:
        files = [path for path in self.root.rglob("*") if path.is_file()]
        sql_files = [path for path in files if path.suffix.lower() == ".sql"]
        reports = [path for path in files if path.suffix.lower() in {".md", ".txt"}]
        databases = [path for path in files if path.suffix.lower() in DATABASE_SUFFIXES]
        return {
            "runtime": "DAComp task-local filesystem",
            "root": str(self.root),
            "file_count": len(files),
            "sql_file_count": len(sql_files),
            "database_files": [str(path.relative_to(self.root)) for path in databases],
            "report_files": [str(path.relative_to(self.root)) for path in reports[-20:]],
            "checkpoint_mode": self.checkpoint_mode,
            "cleanup_events": list(self.cleanup_events[-3:]),
        }

    def stage_data_files(self, paths: tuple[str, ...]) -> dict[str, str]:
        return {Path(path).name: str(Path(path).resolve()) for path in paths}

    def remove_variables(self, variables: tuple[str, ...]) -> None:
        removed: list[str] = []
        ignored: list[str] = []
        root = self.root.resolve()
        for variable in variables:
            if not variable.startswith("file:"):
                ignored.append(variable)
                continue
            relative = variable[5:].strip()
            candidate = (root / relative).resolve()
            if not relative or root not in candidate.parents or not candidate.is_file():
                ignored.append(variable)
                continue
            candidate.unlink()
            removed.append(relative)
        self.cleanup_events.append(
            {"requested": list(variables), "removed_files": removed, "ignored": ignored}
        )

    def file_digest(self, relative_path: str) -> str:
        return hashlib.sha256((self.root / relative_path).read_bytes()).hexdigest()


def _stats(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _fingerprint(root: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple((key, *value) for key, value in sorted(_stats(root).items()))


def _copy_file(source: str, destination: str) -> str:
    """Copy task files, preferring copy-on-write for large benchmark databases."""
    source_path = Path(source)
    if source_path.suffix.lower() in DATABASE_SUFFIXES:
        completed = subprocess.run(
            ["cp", "--reflink=auto", "--", source, destination],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            shutil.copystat(source, destination)
            return destination
    return shutil.copy2(source, destination)
