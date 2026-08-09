from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from stateguard.runtime.executors import ExecutionResult

from .workspace import DACompWorkspace


class DACompProbeExecutor:
    """Fresh manager-only Python scratch with a copied public task directory."""

    def __init__(self, worker_workspace: DACompWorkspace, *, timeout: float = 120.0) -> None:
        self.worker_workspace = worker_workspace
        self.timeout = timeout
        self._relative_files = tuple(
            str(path.relative_to(worker_workspace.source_dir))
            for path in worker_workspace.source_dir.rglob("*")
            if path.is_file()
        )
        self._cache_root: Path | None = None

    def execute(self, code: str) -> ExecutionResult:
        scratch = Path(
            tempfile.mkdtemp(
                prefix="stateguard-dacomp-probe-",
                dir=self.worker_workspace.root.parent,
            )
        )
        try:
            data_root = self._data_root()
            relative_files = repr(self._relative_files)
            bootstrap = (
                "from pathlib import Path\n"
                f"DATA_ROOT = {str(data_root)!r}\n"
                f"_relative_files = {relative_files}\n"
                "data_files = {name: str(Path(DATA_ROOT) / name) "
                "for name in _relative_files}\n"
            )
            completed = subprocess.run(
                [sys.executable, "-c", bootstrap + "\n" + code],
                cwd=scratch,
                env={**os.environ, "PYTHONNOUSERSITE": "1"},
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            output = completed.stdout
            error = completed.stderr if completed.returncode else None
            if completed.stderr:
                output += completed.stderr
            return ExecutionResult(completed.returncode == 0, output, error=error)
        except Exception as exc:
            return ExecutionResult(False, "", error=f"{type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def close(self) -> None:
        if self._cache_root is not None:
            shutil.rmtree(self._cache_root, ignore_errors=True)
            self._cache_root = None

    def _data_root(self) -> Path:
        if self._cache_root is not None:
            return self._cache_root / "task_data"
        self._cache_root = Path(
            tempfile.mkdtemp(
                prefix="stateguard-dacomp-probe-data-",
                dir=self.worker_workspace.root.parent,
            )
        )
        data_root = self._cache_root / "task_data"
        shutil.copytree(
            self.worker_workspace.source_dir,
            data_root,
            copy_function=_copy_probe_file,
        )
        for path in data_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {
                ".duckdb", ".sqlite", ".sqlite3", ".db"
            }:
                path.chmod(0o444)
        return data_root


def _copy_probe_file(source: str, destination: str) -> str:
    source_path = Path(source)
    if source_path.suffix.lower() in {".duckdb", ".sqlite", ".sqlite3", ".db"}:
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
