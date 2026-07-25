from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from stateguard.core.models import to_jsonable


class RunArtifactWriter:
    """Separate worker, manager, repair, and summary traces for reproducibility."""

    def __init__(self, run_dir: Path | None = None) -> None:
        self.run_dir = run_dir
        self.events: dict[str, list[Any]] = {"worker": [], "manager": [], "repair": []}
        if run_dir:
            run_dir.mkdir(parents=True, exist_ok=True)

    def record(self, stream: str, value: Any) -> None:
        if stream not in self.events:
            self.events[stream] = []
        rendered = to_jsonable(value)
        self.events[stream].append(rendered)
        if self.run_dir:
            with (self.run_dir / f"{stream}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rendered, ensure_ascii=False, default=str) + "\n")

    def write_summary(self, value: Any) -> None:
        if self.run_dir:
            _atomic_json(self.run_dir / "summary.json", to_jsonable(value))


def _atomic_json(path: Path, value: Any) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
