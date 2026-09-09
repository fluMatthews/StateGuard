from __future__ import annotations

import copy
import importlib
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from stateguard.harness.engine import (
    StateGuardConfig,
    StateGuardHarness,
    StateGuardResult,
)
from stateguard.runtime.bundle import StateGuardRuntime
from stateguard.runtime.trace import TraceBuffer
from stateguard.telemetry.artifacts import RunArtifactWriter

from .artifacts import write_eval, write_official_artifacts, write_turn_backup
from .dataset import LongDSDataset, LongDSTask
from .evaluator import official_llm_judge
from .executor import LongDSProbeExecutor
from .prompts import load_official_prompts
from .worker import LongDSWorkerAgent
from .workflow import LongDSWorkflow
from .workspace import LongDSWorkspace


@dataclass(frozen=True)
class LongDSRunResult:
    run_dir: Path
    trajectory: dict[str, Any]
    turn_results: tuple[dict[str, Any], ...]
    stateguard_results: tuple[StateGuardResult, ...]


class LongDSAdapter:
    """One-task-at-a-time DSGym integration with optional StateGuard Manager."""

    def __init__(
        self,
        *,
        dsgym_root: Path,
        dataset_root: Path,
        output_root: Path,
        model: str,
        backend_type: str = "litellm",
        manager_url: str = "http://localhost:5000",
        max_steps_per_turn: int = 40,
        temperature: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
        max_model_len: int = 32768,
        worker_max_tokens: int | None = None,
        worker_timeout: float | None = None,
        worker_max_retries: int | None = None,
        reset_env_times: int = 0,
        data_root: Path | None = None,
        backend_factory: Callable[[], Any] | None = None,
        environment_factory: Callable[[], Any] | None = None,
        clean_output: Callable[[list[Any]], str] | None = None,
        system_prompt_template: str | None = None,
    ) -> None:
        self.dsgym_root = dsgym_root.expanduser().resolve()
        self.output_root = output_root.expanduser().resolve()
        self.model = model
        self.backend_type = backend_type
        self.manager_url = manager_url
        self.max_steps_per_turn = max_steps_per_turn
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        self.max_model_len = max_model_len
        self.worker_max_tokens = worker_max_tokens
        self.worker_timeout = worker_timeout
        self.worker_max_retries = worker_max_retries
        self.reset_env_times = reset_env_times
        self.dataset = LongDSDataset(dataset_root, data_root)

        if system_prompt_template is None:
            system_prompt_template, self.judge_prompt = load_official_prompts(
                self.dsgym_root
            )
        else:
            self.judge_prompt = ""
        self.system_prompt_template = system_prompt_template

        self.backend_factory = backend_factory
        self.environment_factory = environment_factory
        self.clean_output = clean_output
        self._runtime_components_lock = threading.Lock()

    def load_tasks(
        self,
        *,
        start_index: int = 0,
        task_limit: int | None = None,
        turn_limit: int | None = None,
    ) -> tuple[LongDSTask, ...]:
        return self.dataset.load(
            start_index=start_index,
            task_limit=task_limit,
            turn_limit=turn_limit,
        )

    def run_task(
        self,
        task: LongDSTask,
        *,
        manager: Any | None = None,
        run_dir: Path | None = None,
    ) -> LongDSRunResult:
        if not task.turns:
            raise ValueError("cannot run an empty LongDS task")
        self._ensure_runtime_components()
        assert self.backend_factory is not None
        assert self.environment_factory is not None
        assert self.clean_output is not None
        run_dir = run_dir or self._run_dir(task)
        state_dir = run_dir / "stateguard"
        artifact_writer = RunArtifactWriter(state_dir)

        started = time.time()
        state_results: list[StateGuardResult] = []
        official_turns: list[dict[str, Any]] = []
        solution_records: list[dict[str, Any]] = []
        total_tokens = 0
        partial_steps = 0
        run_error: str | None = None
        error_type: str | None = None
        error_trace: str | None = None
        cleanup_errors: list[str] = []
        environment: Any | None = None
        worker: LongDSWorkerAgent | None = None
        probe: LongDSProbeExecutor | None = None
        turn_in_progress = False
        try:
            environment = self.environment_factory()
            workspace = LongDSWorkspace(task.data_root)
            worker = LongDSWorkerAgent(
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
                artifacts=artifact_writer,
                manager_probe_executor=probe,
            )
            public_turns = {turn.public.unit_id: turn.public for turn in task.turns}
            workflow = LongDSWorkflow(public_turns, self.system_prompt_template)
            harness = StateGuardHarness(
                runtime=runtime,
                flow_adapter=workflow,
                config=StateGuardConfig(max_worker_steps=self.max_steps_per_turn),
            )

            for turn_number, turn in enumerate(task.turns, 1):
                if (
                    self.reset_env_times > 0
                    and turn_number == self.reset_env_times
                    and worker.messages
                ):
                    worker.reset_environment_preserve_conversation()
                start_message = len(worker.conversation)
                turn_started = time.time()
                turn_in_progress = True
                result = harness.run(turn.public.task_spec())
                turn_in_progress = False
                turn_execution_time = time.time() - turn_started
                state_results.append(result)
                turn_messages = copy.deepcopy(worker.conversation[start_message:])
                successful = result.completed and bool(result.final_answer)
                official_steps = worker.completed_step_count
                total_tokens += worker.turn_token_count
                official_turns.append(
                    {
                        "turn_id": turn.public.turn_id,
                        "question": (
                            f"{turn.public.context}\nQuestion: {turn.public.question}"
                        ),
                        "ground_truth": turn.private.ground_truth,
                        "solution": result.final_answer,
                        "success": successful,
                        "steps": official_steps,
                        "trajectory": turn_messages,
                    }
                )
                solution_records.append(
                    {
                        "turn_id": turn.public.turn_id,
                        "context": turn.public.context,
                        "question": turn.public.question,
                        "ground_truth": turn.private.ground_truth,
                        "solution": result.final_answer,
                        "success": successful,
                        "steps": official_steps,
                        "error": (
                            None
                            if successful
                            else (
                                "Max steps reached"
                                if result.worker_steps >= self.max_steps_per_turn
                                else "No answer"
                            )
                        ),
                        "execution_time": turn_execution_time,
                    }
                )
                write_turn_backup(
                    run_dir,
                    turn_number,
                    copy.deepcopy(worker.conversation),
                )
        except Exception as exc:
            run_error = str(exc)
            error_type = type(exc).__name__
            error_trace = traceback.format_exc()
            if turn_in_progress and worker is not None:
                total_tokens += worker.turn_token_count
                partial_steps = worker.completed_step_count
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

        execution_time = time.time() - started
        conversation = copy.deepcopy(worker.conversation) if worker is not None else []
        metadata = {
            "model": self.model,
            "backend": self.backend_type,
            "max_steps_per_turn": self.max_steps_per_turn,
            "total_tokens": total_tokens,
            "execution_time": execution_time,
            "conversation_length": len(conversation),
            "reset_env_times": self.reset_env_times,
        }
        if run_error is not None:
            metadata.update(
                {
                    "error": run_error,
                    "error_type": error_type,
                    "error_trace": error_trace,
                }
            )
        if cleanup_errors:
            metadata["cleanup_errors"] = cleanup_errors
        trajectory = {
            "solutions": solution_records,
            "success": run_error is None and all(item["success"] for item in official_turns),
            "total_steps": sum(item["steps"] for item in official_turns) + partial_steps,
            "total_turns": len(task.turns),
            "successful_turns": sum(bool(item["success"]) for item in official_turns),
            "error": run_error,
            "metadata": metadata,
            "conversation": conversation,
        }
        write_official_artifacts(
            run_dir=run_dir,
            trajectory=trajectory,
            turn_results=official_turns,
        )
        # Match LONGDS_NO_JUDGE: a no-judge run still has an official-shaped file.
        write_eval(run_dir, copy.deepcopy(official_turns))
        return LongDSRunResult(
            run_dir,
            trajectory,
            tuple(official_turns),
            tuple(state_results),
        )

    def judge(
        self,
        result: LongDSRunResult,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        judge_model: str = "deepseek-v4-pro",
        max_workers: int = 15,
    ) -> list[dict[str, Any]]:
        evaluated = official_llm_judge(
            [copy.deepcopy(item) for item in result.turn_results],
            dsgym_root=self.dsgym_root,
            api_key=api_key,
            base_url=base_url,
            judge_model=judge_model,
            max_workers=max_workers,
        )
        write_eval(result.run_dir, evaluated)
        return evaluated

    def _official_components(
        self,
    ) -> tuple[Callable[[], Any], Callable[[], Any], Callable[[list[Any]], str]]:
        root = str(self.dsgym_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        backends = importlib.import_module("dsgym.agents.backends")
        environment_module = importlib.import_module("dsgym.agents.environment")
        utils = importlib.import_module(
            "dsgym.agents.environment.envs.allocated_code.utils"
        )

        def create_backend() -> Any:
            kwargs: dict[str, Any] = {
                "manager_url": self.manager_url,
                "max_steps": self.max_steps_per_turn,
                "temperature": self.temperature,
                "output_dir": str(self.output_root),
            }
            if self.backend_type in {"vllm", "sglang"}:
                kwargs["max_model_len"] = self.max_model_len
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.worker_timeout is not None:
                kwargs["timeout"] = self.worker_timeout
            if self.worker_max_retries is not None:
                kwargs["max_retries"] = self.worker_max_retries
            backend = backends.get_backend(self.backend_type, self.model, **kwargs)
            if (
                self.backend_type == "litellm"
                and self.worker_max_tokens is not None
                and hasattr(backend, "generation_params")
            ):
                backend.generation_params["max_tokens"] = self.worker_max_tokens
                backend.generation_params["max_completion_tokens"] = (
                    self.worker_max_tokens
                )
            if (
                self.backend_type == "litellm"
                and self.worker_max_retries is not None
                and hasattr(backend, "generation_params")
            ):
                # Avoid nesting LiteLLM retries inside DSGym's backend loop.
                backend.generation_params["num_retries"] = max(
                    0, self.worker_max_retries - 1
                )
            return backend

        def create_environment() -> Any:
            return environment_module.AllocatedCodeEnv(
                manager_url=self.manager_url,
                max_turns=self.max_steps_per_turn,
                output_dir=str(self.output_root),
            )

        return create_backend, create_environment, utils.clean_jupyter_output

    def _ensure_runtime_components(self) -> None:
        if self._runtime_components_ready():
            return
        with self._runtime_components_lock:
            if self._runtime_components_ready():
                return
            official = self._official_components()
            self.backend_factory = self.backend_factory or official[0]
            self.environment_factory = self.environment_factory or official[1]
            self.clean_output = self.clean_output or official[2]

    def _runtime_components_ready(self) -> bool:
        return (
            self.backend_factory is not None
            and self.environment_factory is not None
            and self.clean_output is not None
        )

    def _run_dir(self, task: LongDSTask) -> Path:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        model = self.model.replace("/", "_")
        return (
            self.output_root
            / "longds"
            / task.domain
            / task.dataset_name
            / task.raw_task_id
            / f"{model}_{timestamp}"
        )
