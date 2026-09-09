from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from stateguard.runtime.executors import ExecutionResult

from .workspace import DACompWorkspace


_DATABASE_SUFFIXES = frozenset({".duckdb", ".sqlite", ".sqlite3", ".db"})


# Run inside the failed probe's own scratch, so what it reports is what that
# probe could have reached: the same relative paths, the same database copies.
_LISTING_SNIPPET = r"""
import json as _json
_files = sorted(data_files)
_out = {"files": _files[:60]}
if len(_files) > 60:
    _out["files_omitted"] = len(_files) - 60
try:
    import duckdb as _duckdb
    _tables = {}
    for _name in _files:
        if not _name.lower().endswith((".duckdb", ".db", ".sqlite", ".sqlite3")):
            continue
        try:
            _con = _duckdb.connect(_name, read_only=True)
            _rows = _con.execute(
                "select table_schema || '.' || table_name "
                "from information_schema.tables "
                "where table_schema not in ('information_schema', 'pg_catalog') "
                "order by 1"
            ).fetchall()
            _tables[_name] = [_r[0] for _r in _rows][:80]
            _con.close()
        except Exception as _exc:
            _tables[_name] = "<unreadable: %s>" % type(_exc).__name__
    if _tables:
        _out["tables"] = _tables
except Exception:
    pass
print(_json.dumps(_out, ensure_ascii=False))
"""


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
            self._stage_scratch(data_root, scratch)
            worker_names = self._stage_worker_databases(scratch)
            relative_files = repr(self._relative_files)
            bootstrap = (
                "from pathlib import Path\n"
                f"DATA_ROOT = {str(data_root)!r}\n"
                f"_relative_files = {relative_files}\n"
                "data_files = {name: str(Path(DATA_ROOT) / name) "
                "for name in _relative_files}\n"
                f"_worker_names = {worker_names!r}\n"
                "worker_files = {name: str(Path('worker') / name) "
                "for name in _worker_names}\n"
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
            if error:
                error += self._available_hint(error, scratch, bootstrap)
            return ExecutionResult(completed.returncode == 0, output, error=error)
        except Exception as exc:
            return ExecutionResult(False, "", error=f"{type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _stage_worker_databases(self, scratch: Path) -> tuple[str, ...]:
        """Copy what the Worker has built so far under a worker/ prefix.

        The delivered tree alone cannot settle a claim about the Worker's own
        output: whether staging preserved the row count its source carries takes
        both sides, and one run committed a state asserting counts consistent
        with the contract while every table sat one row short. The copy is per
        probe because the Worker rebuilds between reviews, and it keeps its own
        prefix so an Evol task's delivered sql/ and database stay distinguishable
        from what the Worker has since made of them.
        """
        root = self.worker_workspace.root
        if not root.is_dir():
            return ()
        target = scratch / "worker"
        names: list[str] = []
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _DATABASE_SUFFIXES:
                continue
            target.mkdir(parents=True, exist_ok=True)
            destination = target / path.name
            _copy_probe_file(str(path), str(destination))
            destination.chmod(0o644)
            names.append(path.name)
        return tuple(names)

    _ABSENT_PATH = ("FileNotFoundError", "No such file")
    _ABSENT_TABLE = ("CatalogException", "does not exist")

    def _available_hint(self, error: str, scratch: Path, bootstrap: str) -> str:
        """Name what the scratch holds when a probe addressed something absent.

        A probe that opened a missing path or queried a missing table came back
        as the bare exception, leaving the Manager nothing to correct against:
        one run reissued the same failing snippet twice, and across rounds four
        probes died addressing models the delivered database cannot hold because
        the task asks the Worker to create them. A snippet that never parsed
        gets nothing from here, since a file list cannot help it.
        """
        wants_paths = any(marker in error for marker in self._ABSENT_PATH)
        wants_tables = any(marker in error for marker in self._ABSENT_TABLE)
        if not (wants_paths or wants_tables):
            return ""
        try:
            listing = subprocess.run(
                [sys.executable, "-c", bootstrap + "\n" + _LISTING_SNIPPET],
                cwd=scratch,
                env={**os.environ, "PYTHONNOUSERSITE": "1"},
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            found = json.loads(listing.stdout) if listing.stdout.strip() else {}
        except Exception:
            return ""
        if not isinstance(found, dict):
            return ""
        lines: list[str] = []
        if wants_paths and found.get("files"):
            lines.append("files present: " + ", ".join(found["files"]))
            omitted = found.get("files_omitted")
            if omitted:
                lines.append(f"({omitted} more files not listed)")
        if wants_tables and isinstance(found.get("tables"), dict):
            for name, tables in found["tables"].items():
                rendered = tables if isinstance(tables, str) else ", ".join(tables)
                lines.append(f"tables in {name}: {rendered}")
        if not lines:
            return ""
        return "\n\n[probe workspace] " + "\n".join(lines)

    def _stage_scratch(self, data_root: Path, scratch: Path) -> None:
        """Mirror the task tree into the probe cwd so relative paths resolve.

        DE is the only track whose Worker drives a directory with relative
        paths ("./docs/data_contract.yaml", "asana_start.duckdb"), so the
        Manager copies that idiom into its probes; the other tracks hand the
        Worker flat file handles and their Managers reach for data_files on
        their own. DATA_ROOT and data_files are unchanged -- this only adds the
        relative spelling. Databases are copied writable because duckdb.connect
        opens read-write by default and the cached tree stays 0444; everything
        else is symlinked, so staging costs nothing. The copies live and die
        with this one scratch, so the cached tree cannot be mutated.
        """
        for relative in self._relative_files:
            source = data_root / relative
            destination = scratch / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                continue
            if source.suffix.lower() in _DATABASE_SUFFIXES:
                _copy_probe_file(str(source), str(destination))
                destination.chmod(0o644)
            else:
                destination.symlink_to(source)

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
            if path.is_file() and path.suffix.lower() in _DATABASE_SUFFIXES:
                path.chmod(0o444)
        return data_root


def _copy_probe_file(source: str, destination: str) -> str:
    source_path = Path(source)
    if source_path.suffix.lower() in _DATABASE_SUFFIXES:
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
