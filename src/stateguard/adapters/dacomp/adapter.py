from __future__ import annotations

import re
import importlib
import sys
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from stateguard.harness.engine import StateGuardConfig, StateGuardHarness, StateGuardResult
from stateguard.runtime.bundle import StateGuardRuntime
from stateguard.runtime.trace import TraceBuffer
from stateguard.telemetry.artifacts import RunArtifactWriter

from .artifacts import write_da_stage1_artifacts, write_de_artifacts, write_json
from .dataset import DACompDataset, DACompTask, DACompTrack
from .evaluator import official_da_stage1_rubrics, official_de_cs_cfs
from .executor import DACompProbeExecutor
from .official import official_de_task_spec
from .worker_da import DACompDAWorkerAgent
from .worker_de import DACompDEWorkerAgent
from .worker_de_controller import OpenHandsStepwiseControllerSession
from .workflow import DACompWorkflow
from .workspace import DACompWorkspace


# Each DE track ships a different specification, so the parts of it a probe can
# settle differ. A track absent here renders the lifecycle unchanged.
_CHECK_TARGETS_BY_TRACK = {
    DACompTrack.DE_IMPL: "check_targets_de_impl.txt",
    DACompTrack.DE_EVOL: "check_targets_de_evol.txt",
}


@dataclass(frozen=True)
class DACompRunResult:
    run_dir: Path
    task: DACompTask
    benchmark_result: dict[str, Any]
    stateguard_result: StateGuardResult | None
    error: str | None = None


class DACompAdapter:
    """Official-native DAComp workers with the shared StateGuard manager harness."""

    def __init__(
        self,
        *,
        dacomp_root: Path,
        output_root: Path,
        track: DACompTrack | str,
        model: str,
        experiment_name: str = "default",
        review_cadence: int = 5,
        max_worker_steps: int | None = None,
        da_temperature: float = 0.0,
        da_top_p: float = 1.0,
        da_max_tokens: int = 16384,
        da_max_memory_length: int = 31,
        language: str = "en",
        worker_factory: Callable[[DACompTask, DACompWorkspace], Any] | None = None,
    ) -> None:
        self.root = dacomp_root.expanduser().resolve(strict=True)
        self.output_root = output_root.expanduser().resolve()
        self.track = DACompTrack(track)
        self.model = model
        self.experiment_name = experiment_name
        self.review_cadence = review_cadence
        official_budget = 120 if self.track is DACompTrack.DA_STAGE1 else 30
        self.max_worker_steps = max_worker_steps or official_budget
        self.da_temperature = da_temperature
        self.da_top_p = da_top_p
        self.da_max_tokens = da_max_tokens
        self.da_max_memory_length = da_max_memory_length
        self.language = language
        self.worker_factory = worker_factory
        self.dataset = DACompDataset(self.root)

    def load_tasks(self, **selection: Any) -> tuple[DACompTask, ...]:
        return self.dataset.load(self.track, **selection)

    def run_task(
        self,
        task: DACompTask,
        *,
        manager: Any | None = None,
        run_dir: Path | None = None,
    ) -> DACompRunResult:
        if task.track is not self.track:
            raise ValueError(f"adapter track {self.track.value} cannot run {task.track.value}")
        run_dir = (run_dir or self._run_dir(task)).expanduser().resolve()
        workspace_root = (
            run_dir / "stage1_env"
            if self.track is DACompTrack.DA_STAGE1
            else run_dir
        )
        workspace = DACompWorkspace(workspace_root, task.source_dir)
        # Official DAAgentEnv stages and clears its own mnt_dir. Avoid copying a
        # multi-gigabyte SQLite input twice; injected test workers still receive a
        # prepared filesystem, while DE uses the official copied prediction layout.
        if self.track is not DACompTrack.DA_STAGE1 or self.worker_factory is not None:
            workspace.prepare()
            if self.track is not DACompTrack.DA_STAGE1:
                (workspace.root / "sql").mkdir(parents=True, exist_ok=True)
        else:
            workspace.root.parent.mkdir(parents=True, exist_ok=True)
        writer = RunArtifactWriter(self._artifact_dir(run_dir))
        worker: Any | None = None
        probe: DACompProbeExecutor | None = None
        state_result: StateGuardResult | None = None
        run_error: str | None = None
        error_trace: str | None = None
        benchmark_result: dict[str, Any] = {}
        task_spec = task.task_spec()
        started = time.time()
        try:
            if self.track is not DACompTrack.DA_STAGE1 and self.worker_factory is None:
                task_spec = official_de_task_spec(
                    official_root=self.root, task=task, language=self.language
                )
            worker = self._create_worker(task, workspace)
            probe = DACompProbeExecutor(workspace) if manager is not None else None
            runtime = StateGuardRuntime.create(
                worker=worker,
                manager=manager,
                workspace=workspace,
                trace_buffer=TraceBuffer(),
                artifacts=writer,
                manager_probe_executor=probe,
            )
            harness = StateGuardHarness(
                runtime=runtime,
                flow_adapter=DACompWorkflow(
                    self.review_cadence,
                    hint_includes_variables=self.track is not DACompTrack.DA_STAGE1,
                    check_targets_name=_CHECK_TARGETS_BY_TRACK.get(self.track),
                ),
                config=StateGuardConfig(max_worker_steps=self.max_worker_steps),
            )
            state_result = harness.run(task_spec)
        except Exception as exc:
            run_error = f"{type(exc).__name__}: {exc}"
            error_trace = traceback.format_exc()

        try:
            if self.track is DACompTrack.DA_STAGE1:
                trajectory = worker.trajectory() if worker is not None else {}
                post_process = getattr(worker, "post_process", None)
                result_files = post_process() if callable(post_process) else {}
                benchmark_result = write_da_stage1_artifacts(
                    run_dir=run_dir,
                    workspace_root=workspace.root,
                    official_root=self.root,
                    instance_id=task.instance_id,
                    answer=(state_result.final_answer if state_result else ""),
                    trajectory=trajectory,
                    result_files=result_files,
                    finished=bool(
                        getattr(
                            worker,
                            "official_finished",
                            state_result and state_result.completed,
                        )
                    ),
                    steps=(worker.accepted_steps if worker is not None else 0),
                    error=run_error,
                )
            else:
                run_success, run_output = _validate_de_pipeline(run_dir, worker)
                trajectory = worker.trajectory() if worker is not None else []
                benchmark_result = write_de_artifacts(
                    run_dir=run_dir,
                    instance_id=task.instance_id,
                    instruction=task_spec.query,
                    trajectory=trajectory,
                    run_success=run_success,
                    run_output=run_output,
                    error=run_error,
                )
        except Exception as artifact_exc:
            if run_error is None:
                run_error = f"{type(artifact_exc).__name__}: {artifact_exc}"
                error_trace = traceback.format_exc()
            benchmark_result = {
                "instance_id": task.instance_id,
                "status": "error",
                "error": run_error,
            }
            write_json(run_dir / "result.json", benchmark_result)
        finally:
            if worker is not None:
                try:
                    worker.close()
                except Exception as cleanup_exc:
                    if run_error is None:
                        run_error = f"CleanupError: {cleanup_exc}"
            if probe is not None:
                probe.close()

        write_json(
            run_dir / "run_metadata.json",
            {
                "benchmark": "dacomp",
                "track": self.track.value,
                "instance_id": task.instance_id,
                "model": self.model,
                "manager_enabled": manager is not None,
                "review_cadence": self.review_cadence,
                "worker_budget": self.max_worker_steps,
                "stateguard_artifact_dir": str(writer.run_dir),
                "execution_time": time.time() - started,
                "error": run_error,
                "error_trace": error_trace,
            },
        )
        return DACompRunResult(run_dir, task, benchmark_result, state_result, run_error)

    def judge(self, result: DACompRunResult, **kwargs: Any) -> dict[str, Any]:
        if result.task.track is DACompTrack.DA_STAGE1:
            rubrics_model = kwargs.get("rubrics_model")
            if not rubrics_model:
                raise ValueError("DA stage1 judge requires rubrics_model")
            return official_da_stage1_rubrics(
                official_root=self.root,
                run_dir=result.run_dir,
                instance_id=result.task.instance_id,
                rubrics_model=str(rubrics_model),
                language=str(kwargs.get("language", self.language)),
            )
        return official_de_cs_cfs(
            official_root=self.root,
            prediction_root=result.run_dir.parent,
            instance_id=result.task.instance_id,
            output_dir=result.run_dir / "evaluation",
            python_executable=str(kwargs.get("python_executable") or __import__("sys").executable),
        )

    def _create_worker(self, task: DACompTask, workspace: DACompWorkspace) -> Any:
        if self.worker_factory is not None:
            return self.worker_factory(task, workspace)
        if task.track is DACompTrack.DA_STAGE1:
            return DACompDAWorkerAgent(
                task=task,
                workspace=workspace,
                official_root=self.root,
                model=self.model,
                max_steps=self.max_worker_steps,
                max_tokens=self.da_max_tokens,
                top_p=self.da_top_p,
                temperature=self.da_temperature,
                max_memory_length=self.da_max_memory_length,
                language=self.language,
            )
        session = OpenHandsStepwiseControllerSession(
            official_root=self.root,
            workspace=workspace,
            llm_config_name=self.model,
            max_steps=self.max_worker_steps,
            language=self.language,
        )
        return DACompDEWorkerAgent(session, max_steps=self.max_worker_steps)

    def _run_dir(self, task: DACompTask) -> Path:
        label = _safe_segment(f"{self.model}_{self.experiment_name}")
        return self.output_root / self.track.value / label / task.instance_id

    def _artifact_dir(self, run_dir: Path) -> Path:
        """Keep StateGuard sidecars outside a DE Worker visible workspace."""
        if self.track is DACompTrack.DA_STAGE1:
            # DA is mounted at run_dir/stage1_env, so this is already outside
            # the official Worker workspace.
            return run_dir / "stateguard"
        try:
            relative_run_dir = run_dir.relative_to(self.output_root)
        except ValueError:
            # A caller-supplied run_dir may live outside output_root. Its sibling
            # is still outside the task workspace and cannot pollute ls or find.
            return run_dir.parent / f".{run_dir.name}.stateguard"
        return self.output_root / "_stateguard" / relative_run_dir


def _safe_segment(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return rendered or "default"


def _validate_de_pipeline(run_dir: Path, worker: Any | None = None) -> tuple[bool, str]:
    session = getattr(worker, "session", None)
    runtime = getattr(session, "runtime", None)
    if runtime is not None:
        try:
            actions = importlib.import_module("openhands.events.action")
            check = runtime.run_action(actions.CmdRunAction(
                command="test -f run.py && echo EXISTS || echo MISSING"
            ))
            if "MISSING" in str(getattr(check, "content", "")):
                return False, "run.py file not found"
            result = runtime.run_action(actions.CmdRunAction(
                command="timeout 300 python run.py"
            ))
            content = str(getattr(result, "content", "") or "")
            extras = getattr(result, "extras", {}) or {}
            metadata = extras.get("metadata", {}) if isinstance(extras, dict) else {}
            exit_code = metadata.get("exit_code", 0) if isinstance(metadata, dict) else 0
            return int(exit_code) == 0, content
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
    run_py = run_dir / "run.py"
    if not run_py.is_file():
        return False, "run.py file not found"
    try:
        completed = subprocess.run(
            [sys.executable, "run.py"],
            cwd=run_dir,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, output
