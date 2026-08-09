from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from stateguard.core.events import ReActStep
from stateguard.core.models import AgentAction, Message, TaskSpec, ToolResult, to_jsonable
from stateguard.runtime.workspace import safe_clone

from .dataset import DABstepTask
from .official import OfficialBaselineModules, official_task_prompt


@dataclass(frozen=True)
class DABstepWorkerSnapshot:
    native_logs: Any
    native_state: Any
    executor_state: Any
    executor_custom_tools: Any
    input_messages: Any
    done: bool
    final_answer: str | None


@dataclass(frozen=True)
class DABstepManagerFeedbackLog:
    content: str


class DABstepWorkerAgent:
    """Interruptible facade around the official smolagents 1.3.0 CodeAgent.

    The facade reproduces ``MultiStepAgent.direct_run`` one native code-action at
    a time so the shared harness can pause it. The official prompt, model wrapper,
    parser, persistent Python interpreter, ten-step limit, and max-step fallback
    all remain owned by the downloaded DABstep implementation.
    """

    def __init__(
        self,
        *,
        task: DABstepTask,
        native_agent: Any,
        model_id: str,
        max_steps: int = 10,
        official_modules: OfficialBaselineModules | None = None,
        prompt_builder: Callable[[DABstepTask, Any], str] | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.task = task
        self.native_agent = native_agent
        self.model_id = model_id
        self.max_steps = max_steps
        self.official_modules = official_modules
        self.prompt_builder = prompt_builder
        self._official_memory_writer = native_agent.write_inner_memory_from_logs
        native_agent.write_inner_memory_from_logs = self._write_native_memory
        self._started = False
        self._accepted_steps = 0
        self._trace_step_number = 0
        self._done = False
        self._final_answer: str | None = None

    @property
    def messages(self) -> tuple[Message, ...]:
        if not self._started:
            return ()
        try:
            native = self.native_agent.write_inner_memory_from_logs()
        except Exception:
            native = []
        rendered: list[Message] = []
        for item in native:
            role = getattr(item.get("role", "user"), "value", item.get("role", "user"))
            role = "tool" if str(role) == "tool-response" else str(role)
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            rendered.append(Message(role, str(item.get("content", ""))))
        return tuple(rendered)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def final_answer(self) -> str | None:
        return self._final_answer

    @property
    def accepted_steps(self) -> int:
        return self._accepted_steps

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self._accepted_steps)

    def start(self, prompt: Any, system_prompt: str | None = None) -> None:
        if system_prompt is not None:
            raise ValueError("DABstep uses the official Worker system prompt")
        if self._started:
            raise RuntimeError("DABstep Worker has already started")
        if not isinstance(prompt, TaskSpec):
            raise TypeError("DABstepWorkflow must start the Worker with TaskSpec")
        if prompt.id != self.task.task_id or prompt.query != self.task.question:
            raise ValueError("DABstep TaskSpec does not match the bound official task")
        native_prompt = self._build_prompt()
        self._initialize_native_run(native_prompt)
        self._started = True

    def continue_turn(self, prompt: Any) -> None:
        del prompt
        raise RuntimeError("DABstep is single-query; create a fresh Worker for each task")

    def step(self) -> ReActStep:
        if not self._started:
            raise RuntimeError("DABstep Worker has not started")
        if self._done:
            raise RuntimeError("DABstep Worker is already complete")
        if self._accepted_steps >= self.max_steps:
            raise RuntimeError("DABstep Worker exhausted its official step budget")

        ActionStep, AgentError, AgentMaxStepsError = _native_types()
        native_index = self._accepted_steps
        started = time.time()
        log_entry = ActionStep(step=native_index, start_time=started)
        native_result: Any = None
        unexpected: BaseException | None = None
        try:
            self.native_agent.step_number = native_index
            native_result = self.native_agent.step(log_entry)
        except AgentError as exc:
            # This is exactly how official MultiStepAgent.direct_run records parser,
            # generation, and execution errors before trying its next step.
            log_entry.error = exc
        except BaseException as exc:  # official loop finalizes the log, then propagates
            unexpected = exc
        finally:
            log_entry.end_time = time.time()
            log_entry.duration = log_entry.end_time - started
            self.native_agent.logs.append(log_entry)
            for callback in self.native_agent.step_callbacks:
                callback(log_entry)
            self._accepted_steps += 1
            self.native_agent.step_number = self._accepted_steps
        if unexpected is not None:
            raise unexpected

        forced_final = False
        if native_result is not None:
            self._done = True
            self._final_answer = str(native_result)
        elif self._accepted_steps == self.max_steps:
            # Official DABstep/smolagents performs this additional final-answer
            # model call after max_steps. It is not an 11th code-action step.
            forced_final = True
            final_log = ActionStep(error=AgentMaxStepsError("Reached max steps."))
            self.native_agent.logs.append(final_log)
            fallback = self.native_agent.provide_final_answer(self.native_agent.task)
            final_log.action_output = fallback
            final_log.end_time = time.time()
            final_log.duration = 0
            for callback in self.native_agent.step_callbacks:
                callback(final_log)
            self._done = True
            self._final_answer = str(fallback)

        self._trace_step_number += 1
        return self._to_core_step(log_entry, native_result, forced_final)

    def inject_observation(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        del metadata
        if self._accepted_steps >= self.max_steps:
            raise RuntimeError("cannot repair after the official DABstep budget is exhausted")
        if self._accepted_steps == 0:
            raise RuntimeError("DABstep has no Worker step to receive a repair hint")
        self.native_agent.logs.append(DABstepManagerFeedbackLog(str(content)))
        self._done = False
        self._final_answer = None

    def _write_native_memory(self, summary_mode: bool = False) -> list[dict[str, Any]]:
        """Render official logs plus chronologically placed Manager feedback."""
        memory: list[dict[str, Any]] = []
        segment: list[Any] = []
        for item in self.native_agent.logs:
            if isinstance(item, DABstepManagerFeedbackLog):
                memory.extend(self._render_official_memory_segment(segment, summary_mode))
                segment = []
                if not summary_mode:
                    from smolagents.models import MessageRole

                    memory.append(
                        {
                            "role": MessageRole.USER,
                            "content": (
                                "Additional observation from the external Manager:\n"
                                "<manager_feedback>\n"
                                + item.content
                                + "\n</manager_feedback>"
                            ),
                        }
                    )
            else:
                segment.append(item)
        memory.extend(self._render_official_memory_segment(segment, summary_mode))
        return memory

    def _render_official_memory_segment(
        self, segment: list[Any], summary_mode: bool
    ) -> list[dict[str, Any]]:
        if not segment:
            return []
        complete_logs = self.native_agent.logs
        self.native_agent.logs = segment
        try:
            return self._official_memory_writer(summary_mode=summary_mode)
        finally:
            self.native_agent.logs = complete_logs

    def snapshot(self) -> DABstepWorkerSnapshot:
        executor = self.native_agent.python_executor
        return DABstepWorkerSnapshot(
            native_logs=safe_clone(self.native_agent.logs),
            native_state=safe_clone(self.native_agent.state),
            executor_state=safe_clone(executor.state),
            executor_custom_tools=safe_clone(executor.custom_tools),
            input_messages=safe_clone(self.native_agent.input_messages),
            done=self._done,
            final_answer=self._final_answer,
        )

    def restore(self, snapshot: DABstepWorkerSnapshot) -> None:
        self.native_agent.logs = safe_clone(snapshot.native_logs)
        _replace_mapping(self.native_agent.state, snapshot.native_state)
        executor = self.native_agent.python_executor
        _replace_mapping(executor.state, snapshot.executor_state)
        _replace_mapping(executor.custom_tools, snapshot.executor_custom_tools)
        self.native_agent.input_messages = safe_clone(snapshot.input_messages)
        self._done = snapshot.done
        self._final_answer = snapshot.final_answer
        # Physical calls and trace IDs deliberately remain monotonic. A rollback
        # restores analytical state, not spent official Worker budget.
        self.native_agent.step_number = self._accepted_steps

    def trajectory(self) -> list[dict[str, Any]]:
        return [_render_native_log(item) for item in self.native_agent.logs]

    def close(self) -> None:
        return None

    def _build_prompt(self) -> str:
        if self.prompt_builder is not None:
            return str(self.prompt_builder(self.task, self.native_agent))
        if self.official_modules is None:
            raise RuntimeError("official_modules are required without a prompt_builder")
        return official_task_prompt(
            task=self.task,
            model_id=self.model_id,
            native_agent=self.native_agent,
            modules=self.official_modules,
        )

    def _initialize_native_run(self, prompt: str) -> None:
        from smolagents.agents import SystemPromptStep, TaskStep

        agent = self.native_agent
        agent.task = prompt
        agent.initialize_system_prompt()
        agent.logs = [SystemPromptStep(system_prompt=agent.system_prompt)]
        agent.monitor.reset()
        agent.logs.append(TaskStep(task=prompt))
        agent.step_number = 0

    def _to_core_step(
        self, log_entry: Any, native_result: Any, forced_final: bool
    ) -> ReActStep:
        tool_calls = getattr(log_entry, "tool_calls", None) or []
        code = str(tool_calls[0].arguments) if tool_calls else ""
        error = getattr(log_entry, "error", None)
        observations = str(getattr(log_entry, "observations", "") or "")
        execution_attempted = bool(tool_calls)
        execution_succeeded = execution_attempted and error is None
        tool_result = ToolResult(
            tool_name="python_interpreter",
            ok=execution_succeeded,
            output=observations or (str(error) if error else ""),
            data={
                "execution_attempted": execution_attempted,
                "execution_succeeded": execution_succeeded,
                "code": code,
            },
            error=str(error) if error else None,
        )
        if self._done:
            action = AgentAction(
                kind="final",
                reasoning=str(getattr(log_entry, "llm_output", "") or ""),
                answer=self._final_answer or "",
            )
        else:
            action = AgentAction(
                kind="tool",
                reasoning=str(getattr(log_entry, "llm_output", "") or ""),
                tool_name="python_interpreter",
                arguments={"code": code},
            )
        return ReActStep(
            step_id=self._trace_step_number,
            action=action,
            observation=tool_result,
            done=self._done,
            raw_model_output=str(getattr(log_entry, "llm_output", "") or ""),
            metadata={
                "benchmark": "dabstep",
                "native_step": getattr(log_entry, "step", None),
                "worker_budget_used": self._accepted_steps,
                "worker_budget_limit": self.max_steps,
                "forced_final_after_max_steps": forced_final,
                "native_returned_final": native_result is not None,
                "native_error_type": type(error).__name__ if error else None,
            },
        )


def _native_types() -> tuple[type, type[BaseException], type[BaseException]]:
    from smolagents.agents import ActionStep
    from smolagents.utils import AgentError, AgentMaxStepsError

    return ActionStep, AgentError, AgentMaxStepsError


def _replace_mapping(target: dict[Any, Any], source: Any) -> None:
    target.clear()
    target.update(safe_clone(source))


def _render_native_log(value: Any) -> dict[str, Any]:
    rendered = to_jsonable(value)
    if isinstance(rendered, dict):
        return rendered
    return {"type": type(value).__name__, "value": str(rendered)}
