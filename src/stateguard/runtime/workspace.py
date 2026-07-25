from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class Workspace(Protocol):
    def snapshot(self) -> Any: ...

    def restore(self, snapshot: Any) -> None: ...

    def manifest(self) -> dict[str, Any]: ...

    def stage_data_files(self, paths: tuple[str, ...]) -> dict[str, str]: ...

    def remove_variables(self, variables: tuple[str, ...]) -> None: ...


@dataclass
class InMemoryWorkspace:
    """Small reference workspace; production adapters may use containers/notebooks."""

    variables: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    data_files: dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return safe_clone(
            {
                "variables": self.variables,
                "artifacts": self.artifacts,
                "data_files": self.data_files,
            }
        )

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.variables = safe_clone(snapshot["variables"])
        self.artifacts = safe_clone(snapshot["artifacts"])
        self.data_files = safe_clone(snapshot["data_files"])

    def manifest(self) -> dict[str, Any]:
        return {
            "variables": {name: _summarize(value) for name, value in self.variables.items()},
            "artifacts": {name: _summarize(value) for name, value in self.artifacts.items()},
            "data_files": {
                alias: _file_manifest(Path(path))
                for alias, path in self.data_files.items()
            },
        }

    def stage_data_files(self, paths: tuple[str, ...]) -> dict[str, str]:
        for raw_path in paths:
            source = Path(raw_path).expanduser().resolve(strict=True)
            if not source.is_file():
                raise ValueError(f"task data path is not a file: {source}")
            alias = source.name
            existing = self.data_files.get(alias)
            if existing is not None and existing != str(source):
                raise ValueError(f"duplicate task data filename: {alias}")
            self.data_files[alias] = str(source)
        # The persistent Python executor shares this exact namespace.
        self.variables["data_files"] = dict(self.data_files)
        return dict(self.data_files)

    def remove_variables(self, variables: tuple[str, ...]) -> None:
        for name in variables:
            self.variables.pop(name, None)


@dataclass
class FileWorkspace:
    """Filesystem workspace manifest; snapshotting is supplied by a runtime adapter."""

    root: Path

    def manifest(self) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                files.append({"path": str(path.relative_to(self.root)), "size": path.stat().st_size, "sha256": digest})
        return {"root": str(self.root), "files": files}

    def stage_data_files(self, paths: tuple[str, ...]) -> dict[str, str]:
        self.root.mkdir(parents=True, exist_ok=True)
        staged: dict[str, str] = {}
        for raw_path in paths:
            source = Path(raw_path).expanduser().resolve(strict=True)
            if not source.is_file():
                raise ValueError(f"task data path is not a file: {source}")
            destination = (self.root / source.name).resolve()
            if self.root.resolve() not in destination.parents:
                raise ValueError(f"task data file escapes workspace: {source.name}")
            if source != destination:
                shutil.copy2(source, destination)
            staged[source.name] = str(destination)
        return staged

    def snapshot(self) -> Any:
        raise NotImplementedError("use a container/filesystem checkpoint backend")

    def restore(self, snapshot: Any) -> None:
        raise NotImplementedError("use a container/filesystem checkpoint backend")

    def remove_variables(self, variables: tuple[str, ...]) -> None:
        if variables:
            raise NotImplementedError(
                "FileWorkspace has no language runtime; its adapter must implement "
                "exact variable deletion"
            )


def _summarize(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    if hasattr(value, "shape"):
        summary["shape"] = list(value.shape)
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        rendered = repr(value)
    summary["preview"] = rendered[:500]
    return summary


def _file_manifest(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def safe_clone(value: Any) -> Any:
    """Deep-copy mutable analytical values while tolerating modules/handles."""
    try:
        return copy.deepcopy(value)
    except Exception:
        if isinstance(value, dict):
            return {safe_clone(key): safe_clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [safe_clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(safe_clone(item) for item in value)
        if isinstance(value, set):
            return {safe_clone(item) for item in value}
        # Modules and live runtime handles are retained by identity. They are
        # not mutated by StateGuard rollback; analytical containers around them are.
        return value
