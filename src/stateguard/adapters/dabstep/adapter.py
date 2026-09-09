from __future__ import annotations

import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from stateguard.harness.engine import StateGuardConfig, StateGuardHarness, StateGuardResult
from stateguard.runtime.bundle import StateGuardRuntime
from stateguard.runtime.trace import TraceBuffer
from stateguard.telemetry.artifacts import RunArtifactWriter

from .artifacts import write_config, write_json, write_task_artifacts
from .anthropic import AnthropicMessagesModel
from .dataset import DABstepDataset, DABstepTask
from .evaluator import official_question_score
from .executor import DABstepProbeExecutor
from .official import create_official_agent, load_official_baseline
from .worker import DABstepWorkerAgent
from .workflow import DABstepWorkflow
from .workspace import DABstepWorkspace


@dataclass(frozen=True)
class DABstepRunResult:
    run_dir: Path
    task: DABstepTask
    benchmark_result: dict[str, Any]
    stateguard_result: StateGuardResult | None
    error: str | None = None


class DABstepAdapter:
    """Official DABstep Worker runtime with an optional external Manager."""

    def __init__(
        self,
        *,
        dabstep_root: Path,
        output_root: Path,
        model_id: str,
        split: str = "default",
        experiment_name: str = "default",
        official_runner_root: Path | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        max_worker_steps: int = 10,
        review_cadence: int = 3,
        runtime_profile: str = "compat-v1",
        worker_factory: Callable[[DABstepTask, DABstepWorkspace], Any] | None = None,
    ) -> None:
        if max_worker_steps < 1:
            raise ValueError("max_worker_steps must be positive")
        if runtime_profile not in {"compat-v1", "official"}:
            raise ValueError("runtime_profile must be compat-v1 or official")
        self.root = dabstep_root.expanduser().resolve(strict=True)
        self.official_runner_root = (
            official_runner_root or self.root / "dabstep_official_runner"
        ).expanduser().resolve(strict=True)
        self.output_root = output_root.expanduser().resolve()
        self.model_id = model_id
        self.split = split
        self.experiment_name = experiment_name
        self.api_base = api_base
        self.api_key = api_key
        self.max_worker_steps = max_worker_steps
        self.review_cadence = review_cadence
        self.runtime_profile = runtime_profile
        self.worker_factory = worker_factory
        if worker_factory is None:
            from .runtime_compat import configure_smolagents_runtime

            self.runtime_report = configure_smolagents_runtime(runtime_profile)
        else:
            # Test/custom workers do not use the official smolagents runtime.
            self.runtime_report = {
                "runtime_profile": runtime_profile,
                "configured": False,
                "reason": "custom worker_factory",
            }
        self.dataset = DABstepDataset(self.root)
        self.run_root = (
            self.output_root
            / _safe_segment(self.model_id)
            / self.split
            / _safe_segment(self.experiment_name)
        )
        self.run_root.mkdir(parents=True, exist_ok=True)
        write_config(
            self.run_root / "config.yaml",
            {
                "model_id": self.model_id,
                "split": self.split,
                "experiment": self.experiment_name,
                "max_steps": self.max_worker_steps,
                "review_cadence": self.review_cadence,
                "runtime_profile": self.runtime_profile,
                "runtime_report": self.runtime_report,
                "official_runner_root": str(self.official_runner_root),
            },
        )

    def load_tasks(self, **selection: Any) -> tuple[DABstepTask, ...]:
        return self.dataset.load(self.split, **selection)

    def run_task(
        self,
        task: DABstepTask,
        *,
        manager: Any | None = None,
        run_dir: Path | None = None,
    ) -> DABstepRunResult:
        if task.split != self.split:
            raise ValueError(f"adapter split {self.split} cannot run task split {task.split}")
        run_dir = (run_dir or self.run_root / "tasks" / task.task_id).resolve()
        workspace = DABstepWorkspace(task.context_dir)
        writer = RunArtifactWriter(run_dir / "stateguard")
        worker: Any | None = None
        probe: DABstepProbeExecutor | None = None
        state_result: StateGuardResult | None = None
        run_error: str | None = None
        error_trace: str | None = None
        started = time.time()
        try:
            worker = self._create_worker(task, workspace)
            workspace.bind_worker(worker)
            probe = (
                DABstepProbeExecutor(workspace, runtime_profile=self.runtime_profile)
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
            harness = StateGuardHarness(
                runtime=runtime,
                flow_adapter=DABstepWorkflow(self.review_cadence),
                config=StateGuardConfig(max_worker_steps=self.max_worker_steps),
            )
            state_result = harness.run(task.task_spec())
        except Exception as exc:
            run_error = f"{type(exc).__name__}: {exc}"
            error_trace = traceback.format_exc()

        answer = (
            state_result.final_answer
            if state_result is not None
            else str(getattr(worker, "final_answer", "") or "")
        )
        answer_entry: dict[str, Any] = {
            "task_id": task.task_id,
            "agent_answer": answer,
        }
        if task.split == "dev":
            if task.reference_answer is None:
                raise ValueError("dev task is missing evaluator-only reference answer")
            answer_entry.update(
                {
                    "answer": task.reference_answer,
                    "score": official_question_score(
                        official_runner_root=self.official_runner_root,
                        agent_answer=answer,
                        reference_answer=task.reference_answer,
                    ),
                    "level": task.level,
                }
            )
        trajectory = worker.trajectory() if worker is not None else []
        benchmark_result = write_task_artifacts(
            run_root=self.run_root,
            task_dir=run_dir,
            answer_entry=answer_entry,
            trajectory=trajectory,
            error=run_error,
        )
        write_json(
            run_dir / "run_metadata.json",
            {
                "benchmark": "dabstep",
                "task_id": task.task_id,
                "split": task.split,
                "level": task.level,
                "model_id": self.model_id,
                "manager_enabled": manager is not None,
                "review_cadence": self.review_cadence,
                "worker_budget": self.max_worker_steps,
                "runtime_profile": self.runtime_profile,
                "runtime_report": self.runtime_report,
                "stateguard_artifact_dir": str(writer.run_dir),
                "execution_time": time.time() - started,
                "error": run_error,
                "error_trace": error_trace,
            },
        )
        if worker is not None:
            try:
                worker.close()
            except Exception:
                pass
        if probe is not None:
            probe.close()
        return DABstepRunResult(run_dir, task, benchmark_result, state_result, run_error)

    def judge(self, result: DABstepRunResult) -> dict[str, Any]:
        if result.task.reference_answer is None:
            raise ValueError("official local scoring is available only for the dev split")
        score = official_question_score(
            official_runner_root=self.official_runner_root,
            agent_answer=str(result.benchmark_result.get("agent_answer", "")),
            reference_answer=result.task.reference_answer,
        )
        return {
            "task_id": result.task.task_id,
            "score": score,
            "level": result.task.level,
            "agent_answer": result.benchmark_result.get("agent_answer", ""),
        }

    def _create_worker(self, task: DABstepTask, workspace: DABstepWorkspace) -> Any:
        if self.worker_factory is not None:
            return self.worker_factory(task, workspace)
        modules = load_official_baseline(self.official_runner_root)
        native_agent = create_official_agent(
            modules=modules,
            model_id=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            max_steps=self.max_worker_steps,
            context_dir=task.context_dir,
        )
        if self.model_id.startswith("anthropic/"):
            if not self.api_base or not self.api_key:
                raise ValueError("Anthropic Worker requires api_base and api_key")
            native_agent.model = AnthropicMessagesModel(
                model_id=self.model_id,
                api_base=self.api_base,
                api_key=self.api_key,
                max_tokens=3000,
            )
        return DABstepWorkerAgent(
            task=task,
            native_agent=native_agent,
            model_id=self.model_id,
            max_steps=self.max_worker_steps,
            official_modules=modules,
        )


def _safe_segment(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return rendered or "default"
