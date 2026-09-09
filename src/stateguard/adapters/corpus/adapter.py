from __future__ import annotations

import copy
import filecmp
import re
import shutil
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from stateguard.adapters.longds.adapter import LongDSAdapter
from stateguard.adapters.longds.executor import LongDSProbeExecutor
from stateguard.adapters.longds.workspace import LongDSWorkspace
from stateguard.core.models import to_jsonable
from stateguard.harness.engine import StateGuardConfig, StateGuardHarness, StateGuardResult
from stateguard.runtime.bundle import StateGuardRuntime
from stateguard.runtime.trace import TraceBuffer
from stateguard.sft.activation import export_manager_activations
from stateguard.telemetry.artifacts import RunArtifactWriter

from .artifacts import write_json
from .dataset import CorpusMode, CorpusTask, CorpusUnit, create_corpus_loader
from .worker import CorpusWorkerAgent
from .workflow import CorpusMultiTurnWorkflow, CorpusSingleQueryWorkflow


@dataclass(frozen=True)
class CorpusRunResult:
    run_dir: Path
    task: CorpusTask
    trajectory: dict[str, Any]
    unit_results: tuple[dict[str, Any], ...]
    stateguard_results: tuple[StateGuardResult, ...]
    error: str | None = None


class CorpusAdapter(LongDSAdapter):
    """Run normalized corpora with the tested DSGym Worker and StateGuard core."""

    def __init__(
        self,
        *,
        source: str,
        corpus_root: Path,
        dsgym_root: Path,
        output_root: Path,
        model: str,
        experiment_name: str = "default",
        mode: CorpusMode | None = None,
        review_cadence: int = 3,
        backend_type: str = "litellm",
        manager_url: str = "http://localhost:5000",
        max_worker_steps: int = 30,
        temperature: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
        max_model_len: int = 32768,
        worker_max_tokens: int | None = None,
        worker_timeout: float | None = None,
        worker_max_retries: int | None = None,
        backend_factory: Callable[[], Any] | None = None,
        environment_factory: Callable[[], Any] | None = None,
        clean_output: Callable[[list[Any]], str] | None = None,
        system_prompt_template: str | None = None,
    ) -> None:
        if review_cadence < 1:
            raise ValueError("review_cadence must be positive")
        super().__init__(
            dsgym_root=dsgym_root,
            # The inherited object supplies only the official Worker runtime;
            # corpus loading is replaced below and never calls LongDSDataset.load.
            dataset_root=corpus_root,
            output_root=output_root,
            model=model,
            backend_type=backend_type,
            manager_url=manager_url,
            max_steps_per_turn=max_worker_steps,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            max_model_len=max_model_len,
            worker_max_tokens=worker_max_tokens,
            worker_timeout=worker_timeout,
            worker_max_retries=worker_max_retries,
            backend_factory=backend_factory,
            environment_factory=environment_factory,
            clean_output=clean_output,
            system_prompt_template=system_prompt_template,
        )
        self.source = source.strip().lower().replace("-", "_")
        self.corpus_root = corpus_root.expanduser().resolve(strict=True)
        self.corpus_loader = create_corpus_loader(self.source, self.corpus_root)
        self.mode = mode
        self.review_cadence = review_cadence
        self.experiment_name = experiment_name
        self.run_root = (
            self.output_root
            / "corpus"
            / self.source
            / _safe_segment(self.model)
            / _safe_segment(experiment_name)
        )
        self.run_root.mkdir(parents=True, exist_ok=True)

    def load_tasks(
        self,
        *,
        task_ids: tuple[str, ...] | None = None,
        start_index: int = 0,
        task_limit: int | None = None,
        unit_limit: int | None = None,
    ) -> tuple[CorpusTask, ...]:
        return self.corpus_loader.load(
            mode=self.mode,
            task_ids=task_ids,
            start_index=start_index,
            task_limit=task_limit,
            unit_limit=unit_limit,
        )

    def run_task(
        self,
        task: CorpusTask,
        *,
        manager: Any | None = None,
        run_dir: Path | None = None,
    ) -> CorpusRunResult:
        if not task.units:
            raise ValueError("cannot run an empty corpus task")
        if self.mode is not None and task.mode != self.mode:
            raise ValueError(f"adapter mode {self.mode} cannot run {task.mode}")
        self._ensure_runtime_components()
        assert self.backend_factory is not None
        assert self.environment_factory is not None
        assert self.clean_output is not None

        run_dir = (run_dir or self._run_dir(task)).resolve()
        writer = RunArtifactWriter(run_dir / "stateguard")
        runtime_task = self._runtime_task(task, run_dir)
        started = time.time()
        environment: Any | None = None
        worker: CorpusWorkerAgent | None = None
        probe: LongDSProbeExecutor | None = None
        state_results: list[StateGuardResult] = []
        unit_results: list[dict[str, Any]] = []
        manager_failures: list[dict[str, Any]] = []
        run_error: str | None = None
        error_type: str | None = None
        error_trace: str | None = None
        cleanup_errors: list[str] = []

        try:
            environment = self.environment_factory()
            workspace = LongDSWorkspace(runtime_task.data_root)
            worker = CorpusWorkerAgent(
                backend=self.backend_factory(),
                environment=environment,
                workspace=workspace,
                clean_output=self.clean_output,
                max_steps_per_turn=self.max_steps_per_turn,
            )
            probe = (
                LongDSProbeExecutor(workspace, self.environment_factory, self.clean_output)
                if manager is not None
                else None
            )
            runtime = StateGuardRuntime.create(
                worker=worker,
                manager=manager,
                workspace=workspace,
                trace_buffer=TraceBuffer(),
                artifacts=writer,
                manager_probe_executor=probe,
            )
            if runtime_task.mode == "multi_turn":
                public_units = {
                    unit.public.unit_id: unit.public for unit in runtime_task.units
                }
                workflow: Any = CorpusMultiTurnWorkflow(
                    public_units, self.system_prompt_template
                )
            else:
                workflow = CorpusSingleQueryWorkflow(
                    runtime_task.units[0].public,
                    self.system_prompt_template,
                    review_cadence=self.review_cadence,
                )
            harness = StateGuardHarness(
                runtime=runtime,
                flow_adapter=workflow,
                config=StateGuardConfig(max_worker_steps=self.max_steps_per_turn),
            )

            for unit in runtime_task.units:
                message_start = len(worker.conversation)
                unit_started = time.time()
                result = harness.run(unit.public.task_spec())
                state_results.append(result)
                manager_failures.extend(
                    {
                        **to_jsonable(failure),
                        "task_id": unit.public.unit_id,
                    }
                    for failure in result.manager_failures
                )
                # An empty submitted solution is reviewable Manager evidence, not
                # a failed unit by itself. Exclude it only when the Worker actually
                # consumed the whole per-unit budget without producing an answer.
                worker_budget_exhausted = bool(
                    not result.final_answer
                    and result.worker_steps >= self.max_steps_per_turn
                )
                sft_eligible = bool(
                    result.completed
                    and not result.degraded
                    and not worker_budget_exhausted
                )
                unit_results.append(
                    {
                        "unit_id": unit.public.unit_id,
                        "unit_index": unit.public.unit_index,
                        "query": unit.public.query,
                        "context": unit.public.context,
                        "solution": result.final_answer,
                        "completed": result.completed,
                        "worker_steps": result.worker_steps,
                        "manager_actions": result.manager_actions,
                        "repair_count": result.repair_count,
                        "degraded": result.degraded,
                        "worker_budget_exhausted": worker_budget_exhausted,
                        "sft_eligible": sft_eligible,
                        "execution_time": time.time() - unit_started,
                        "trajectory": copy.deepcopy(worker.conversation[message_start:]),
                    }
                )
        except Exception as exc:
            run_error = str(exc)
            error_type = type(exc).__name__
            error_trace = traceback.format_exc()
        finally:
            if worker is not None:
                try:
                    worker.close()
                except Exception as exc:
                    cleanup_errors.append(f"worker.close: {type(exc).__name__}: {exc}")
            elif environment is not None:
                try:
                    environment.close()
                except Exception as exc:
                    cleanup_errors.append(f"environment.close: {type(exc).__name__}: {exc}")
            if probe is not None:
                try:
                    probe.close()
                except Exception as exc:
                    cleanup_errors.append(f"probe.close: {type(exc).__name__}: {exc}")

        if run_error is None and cleanup_errors:
            run_error = cleanup_errors[0]
            error_type = "CleanupError"
        conversation = copy.deepcopy(worker.conversation) if worker is not None else []
        trajectory = {
            "source": task.source,
            "mode": task.mode,
            "task_id": task.raw_task_id,
            "model": self.model,
            "success": (
                run_error is None
                and len(unit_results) == len(task.units)
                and all(row["completed"] and row["solution"] for row in unit_results)
            ),
            "completed_units": len(unit_results),
            "total_units": len(task.units),
            "total_worker_steps": sum(row["worker_steps"] for row in unit_results),
            "execution_time": time.time() - started,
            "conversation": conversation,
            "units": unit_results,
            "error": run_error,
            "error_type": error_type,
            "error_trace": error_trace,
            "cleanup_errors": cleanup_errors,
        }
        write_json(run_dir / "trajectory.json", trajectory)
        write_json(
            run_dir / "private" / "references.json",
            {
                "warning": "Evaluator-only. Never load into Worker or Manager runtime.",
                "units": [
                    {
                        "unit_id": unit.public.unit_id,
                        "reference_answer": unit.private.reference_answer,
                        "expected_relation": unit.private.expected_relation,
                        "expected_upstream": unit.private.expected_upstream,
                        "provenance": unit.private.provenance,
                    }
                    for unit in task.units
                ],
            },
        )
        write_json(
            run_dir / "run_metadata.json",
            {
                "source": task.source,
                "mode": task.mode,
                "task_id": task.raw_task_id,
                "model": self.model,
                "manager_enabled": manager is not None,
                "worker_budget_per_unit": self.max_steps_per_turn,
                "review_cadence": (
                    self.review_cadence if task.mode == "single_query" else None
                ),
                "manager_steps_count_toward_worker_budget": False,
                "data_root": str(runtime_task.data_root),
                "stateguard_artifact_dir": str(writer.run_dir),
                "error": run_error,
            },
        )
        # Single-query flow has exactly one unit, so any unit-level gate turns
        # the whole task into an all-or-nothing export. Manager decisions are
        # judged per activation there: the activation-level filter already
        # rejects a failing or malformed block on its own, and a Worker that
        # never reached an answer does not make the Manager's own state
        # handling wrong. Multi-turn keeps the per-turn gate, where a failed
        # turn really does leave its state unwritten.
        eligible_unit_ids = (
            None
            if task.mode == "single_query"
            else {
                str(row["unit_id"])
                for row in unit_results
                if row["sft_eligible"]
            }
        )
        self._write_manager_and_sft(
            run_dir=run_dir,
            manager=manager,
            manager_failures=manager_failures,
            eligible_unit_ids=eligible_unit_ids,
        )
        return CorpusRunResult(
            run_dir,
            task,
            trajectory,
            tuple(unit_results),
            tuple(state_results),
            run_error,
        )

    def _runtime_task(self, task: CorpusTask, run_dir: Path) -> CorpusTask:
        """Expose declared data files without exposing corpus annotations.

        Every corpus is staged into a run-local, data-only directory. This keeps
        question files, answers, relation annotations, and ground-truth artifacts
        outside the path shown to either agent. Relative subdirectories are kept so
        that benchmark-style data paths continue to work.
        """
        data_root = run_dir / "worker_data"
        data_root.mkdir(parents=True, exist_ok=True)
        source_root = task.data_root.resolve(strict=True)
        staged: dict[Path, Path] = {}
        sources = {
            path.resolve(strict=True)
            for unit in task.units
            for path in unit.public.data_files
        }
        for source in sorted(sources):
            try:
                relative = source.relative_to(source_root)
            except ValueError as exc:
                raise ValueError(
                    f"declared corpus data is outside its data root: {source}"
                ) from exc
            destination = data_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _stage_data_file(source, destination)
            staged[source] = destination.resolve(strict=True)
        units = tuple(
            CorpusUnit(
                replace(
                    unit.public,
                    data_root=data_root,
                    data_files=tuple(
                        staged[path.resolve(strict=True)]
                        for path in unit.public.data_files
                    ),
                ),
                unit.private,
            )
            for unit in task.units
        )
        return CorpusTask(task.source, task.raw_task_id, task.mode, units)

    def _write_manager_and_sft(
        self,
        *,
        run_dir: Path,
        manager: Any | None,
        manager_failures: list[dict[str, Any]],
        eligible_unit_ids: set[str] | None = None,
    ) -> None:
        export_session = getattr(manager, "export_session", None)
        if manager is None or not callable(export_session):
            return
        session = export_session()
        write_json(run_dir / "manager_session.json", session)
        write_json(run_dir / "manager_failures.json", manager_failures)
        messages = session.get("messages") if isinstance(session, dict) else None
        if not isinstance(messages, list) or len(messages) < 3:
            write_json(
                run_dir / "sft_export_report.json",
                {
                    "status": "skipped_no_activations",
                    "accepted": 0,
                    "rejected": 0,
                },
            )
            return
        try:
            records, report = export_manager_activations(
                session,
                manager_failures,
                source=str(run_dir / "manager_session.json"),
                eligible_task_ids=eligible_unit_ids,
            )
            write_json(run_dir / "sft_data_activations.json", records)
            write_json(run_dir / "sft_export_report.json", report)
        except Exception as exc:
            write_json(
                run_dir / "sft_export_report.json",
                {
                    "export_error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )

    def _run_dir(self, task: CorpusTask) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.run_root / task.mode / task.raw_task_id / timestamp


def _safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()) or "default"


def _stage_data_file(source: Path, destination: Path) -> None:
    """Create one safe, repeatable staged data file."""
    if destination.is_symlink():
        raise FileExistsError(f"staged data path cannot be a symlink: {destination}")
    if destination.exists():
        if not destination.is_file():
            raise FileExistsError(f"staged data path is not a file: {destination}")
        try:
            same_file = destination.samefile(source)
        except OSError:
            same_file = False
        if same_file or filecmp.cmp(source, destination, shallow=False):
            return
        raise FileExistsError(f"staged data collision: {destination}")
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)

