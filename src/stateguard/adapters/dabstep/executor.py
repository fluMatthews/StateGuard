from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from stateguard.runtime.executors import ExecutionResult

from .workspace import DABstepWorkspace


_PROBE_BOOTSTRAP = r"""
import base64
import json
import sys

from smolagents.local_python_executor import LocalPythonInterpreter
from stateguard.adapters.dabstep.runtime_compat import configure_smolagents_runtime

def decode(value):
    return base64.b64decode(value.encode("ascii")).decode("utf-8")

def read_only_open(*args, **kwargs):
    if (len(args) > 1 and isinstance(args[1], str) and args[1] != "r") or kwargs.get("mode", "r") != "r":
        raise Exception("Only mode=\"r\" allowed for the function open")
    return open(*args, **kwargs)

profile, encoded_code, encoded_imports, encoded_files, encoded_root = sys.argv[1:]
configure_smolagents_runtime(profile)
code = decode(encoded_code)
authorized_imports = json.loads(decode(encoded_imports))
data_files = json.loads(decode(encoded_files))
data_root = decode(encoded_root)
interpreter = LocalPythonInterpreter(authorized_imports, tools={"open": read_only_open})
result, logs, _ = interpreter(
    code,
    {"DATA_ROOT": data_root, "data_files": data_files},
)
if logs:
    print(logs, end="" if logs.endswith("\n") else "\n")
if result is not None:
    print(repr(result))
"""

def _encode(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


class DABstepProbeExecutor:
    """Short-lived Manager-only Python scratch over read-only public context."""

    def __init__(
        self,
        worker_workspace: DABstepWorkspace,
        *,
        runtime_profile: str = "compat-v1",
        timeout: float = 120.0,
    ) -> None:
        self.worker_workspace = worker_workspace
        self.runtime_profile = runtime_profile
        self.timeout = timeout
        self._files = tuple(path.name for path in worker_workspace._files)

    def execute(self, code: str) -> ExecutionResult:
        scratch = Path(tempfile.mkdtemp(prefix="stateguard-dabstep-probe-"))
        try:
            worker_executor = self.worker_workspace._executor()
            authorized_imports = list(worker_executor.additional_authorized_imports)
            data_files = {
                name: str(self.worker_workspace.root / name) for name in self._files
            }
            arguments = (
                self.runtime_profile,
                _encode(code),
                _encode(json.dumps(authorized_imports)),
                _encode(json.dumps(data_files)),
                _encode(str(self.worker_workspace.root)),
            )
            completed = subprocess.run(
                [sys.executable, "-c", _PROBE_BOOTSTRAP, *arguments],
                cwd=scratch,
                env={**os.environ, "PYTHONNOUSERSITE": "1"},
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            return ExecutionResult(
                completed.returncode == 0,
                output,
                error=completed.stderr if completed.returncode else None,
            )
        except Exception as exc:
            return ExecutionResult(False, "", error=f"{type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def close(self) -> None:
        return None
