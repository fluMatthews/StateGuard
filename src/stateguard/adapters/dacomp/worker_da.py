from __future__ import annotations

import copy
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stateguard.core.events import ReActStep
from stateguard.core.models import AgentAction, Message, ToolResult

from .dataset import DACompTask
from .workspace import DACompWorkspace


@dataclass(frozen=True)
class DAWorkerSnapshot:
    agent_state: dict[str, Any]
    observation: str
    accepted_steps: int
    parse_retries: int
    done: bool
    official_finished: bool
    final_answer: str | None


class DACompDAWorkerAgent:
    """Interruptible wrapper around the official DA stage-1 PromptAgent/DAAgentEnv.

    One StateGuard Worker step is one successfully parsed official action. Invalid
    generations are retried internally and do not consume the official 120-step budget,
    exactly matching ``PromptAgent.run``.
    """

    def __init__(
        self,
        *,
        task: DACompTask,
        workspace: DACompWorkspace,
        official_root: Path,
        model: str,
        max_steps: int = 120,
        max_tokens: int = 16384,
        top_p: float = 1.0,
        temperature: float = 0.0,
        max_memory_length: int = 31,
        language: str = "en",
        agent: Any | None = None,
        environment: Any | None = None,
    ) -> None:
        self.task = task
        self.workspace = workspace
        self.max_steps = max_steps
        self._observation = "You are in the folder now."
        self._accepted_steps = 0
        self._parse_retries = 0
        self._done = False
        # ``_done`` is the facade stop flag used by the generic harness.  The
        # official PromptAgent, however, reports ``finished=False`` when its
        # action budget or parse-retry budget is exhausted without Terminate.
        self._official_finished = False
        self._final_answer: str | None = None
        self._result_files: dict[str, Any] | None = None
        if agent is None or environment is None:
            agent, environment = self._create_official_components(
                official_root=official_root,
                model=model,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                max_memory_length=max_memory_length,
                language=language,
            )
        self.agent = agent
        self.environment = environment
        self.agent.set_env_and_task(self.environment)

    @property
    def messages(self) -> tuple[Message, ...]:
        rendered: list[Message] = []
        for item in self.agent.history_messages:
            content = item.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            rendered.append(Message(str(item.get("role", "user")), str(content)))
        return tuple(rendered)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def final_answer(self) -> str | None:
        return self._final_answer

    @property
    def official_finished(self) -> bool:
        """Whether the official DA environment actually emitted Terminate."""
        return self._official_finished

    @property
    def accepted_steps(self) -> int:
        return self._accepted_steps

    def start(self, prompt: Any, system_prompt: str | None = None) -> None:
        if system_prompt is not None:
            raise ValueError("DA stage1 uses the official PromptAgent system message")
        if str(prompt).strip() != self.task.instruction.strip():
            raise ValueError("DA stage1 prompt must equal the official task instruction")
        if self._accepted_steps or self._done:
            raise RuntimeError("DA Worker has already started")

    def continue_turn(self, prompt: Any) -> None:
        del prompt
        raise RuntimeError("DAComp is single-query; a Worker cannot continue into another task")

    def step(self) -> ReActStep:
        if self._done:
            raise RuntimeError("DA stage1 Worker is already complete")
        if self._accepted_steps >= self.max_steps:
            self._done = True
            raise RuntimeError("DA stage1 Worker exhausted its official action budget")

        response = ""
        action = None
        while action is None:
            response, action = self.agent.predict(self._observation)
            if action is not None:
                break
            self._parse_retries += 1
            if self._parse_retries > 40:
                # Official PromptAgent.run returns ``done=False, result=""``
                # here; it does not raise. Stop only the facade loop and retain
                # that official outcome in the benchmark artifact.
                self._done = True
                self._final_answer = ""
                return ReActStep(
                    step_id=self._accepted_steps + 1,
                    action=AgentAction(
                        kind="final",
                        reasoning="Official PromptAgent stopped after 40 parse retries.",
                        answer="",
                    ),
                    observation=None,
                    done=True,
                    raw_model_output=response,
                    metadata={
                        "benchmark": "dacomp",
                        "track": "da-stage1",
                        "official_action": "",
                        "official_observation": self._observation,
                        "official_stop_reason": "parse_retry_exhausted",
                        "worker_budget_used": self._accepted_steps,
                        "worker_budget_limit": self.max_steps,
                        "parse_retries": self._parse_retries,
                    },
                )
            self._observation = _parse_retry_observation(response)

        accepted_observation, terminal = self.environment.step(action)
        self._accepted_steps += 1
        self._observation = str(accepted_observation)
        action_name = type(action).__name__
        thought = str(self.agent.thoughts[-1]) if self.agent.thoughts else ""
        official_action = str(action)

        if terminal:
            self._done = True
            self._official_finished = True
            output = getattr(action, "output", "")
            self._final_answer = str(output or "")
            core_action = AgentAction(
                kind="final", reasoning=thought, answer=self._final_answer
            )
            tool_result = None
        else:
            succeeded = _observation_succeeded(self._observation)
            core_action = AgentAction(
                kind="tool",
                reasoning=thought,
                tool_name=action_name,
                arguments=_public_action_arguments(action),
            )
            tool_result = ToolResult(
                tool_name=action_name,
                ok=succeeded,
                output=self._observation,
                data={
                    "execution_attempted": True,
                    "execution_succeeded": succeeded,
                    "native_action": official_action,
                },
                error=None if succeeded else self._observation,
            )
            if self._accepted_steps >= self.max_steps:
                self._done = True
                self._final_answer = ""

        return ReActStep(
            step_id=self._accepted_steps,
            action=core_action,
            observation=tool_result,
            done=self._done,
            raw_model_output=response,
            metadata={
                "benchmark": "dacomp",
                "track": "da-stage1",
                "official_action": official_action,
                "official_observation": self._observation,
                "worker_budget_used": self._accepted_steps,
                "worker_budget_limit": self.max_steps,
                "parse_retries": self._parse_retries,
            },
        )

    def inject_observation(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        del metadata
        self._observation = self._observation + "\n\n" + str(content)
        # A repair after Terminate continues the same append-only official history.
        if self._done and self._accepted_steps < self.max_steps:
            self._done = False
            self._official_finished = False
            self._final_answer = None

    def snapshot(self) -> DAWorkerSnapshot:
        fields = (
            "thoughts",
            "responses",
            "actions",
            "observations",
            "history_messages",
            "codes",
            "_last_repetition_signature",
        )
        return DAWorkerSnapshot(
            {name: copy.deepcopy(getattr(self.agent, name)) for name in fields},
            self._observation,
            self._accepted_steps,
            self._parse_retries,
            self._done,
            self._official_finished,
            self._final_answer,
        )

    def restore(self, snapshot: DAWorkerSnapshot) -> None:
        for name, value in snapshot.agent_state.items():
            setattr(self.agent, name, copy.deepcopy(value))
        self._observation = snapshot.observation
        self._accepted_steps = snapshot.accepted_steps
        self._parse_retries = snapshot.parse_retries
        self._done = snapshot.done
        self._official_finished = snapshot.official_finished
        self._final_answer = snapshot.final_answer

    def trajectory(self) -> dict[str, Any]:
        return self.agent.get_trajectory()

    def post_process(self) -> dict[str, Any]:
        if self._result_files is None:
            self._result_files = self.environment.post_process()
        return copy.deepcopy(self._result_files)

    def close(self) -> None:
        self.environment.close()

    def _create_official_components(
        self,
        *,
        official_root: Path,
        model: str,
        max_tokens: int,
        top_p: float,
        temperature: float,
        max_memory_length: int,
        language: str,
    ) -> tuple[Any, Any]:
        method_root = official_root / "methods" / "da-agent"
        root_string = str(method_root.resolve(strict=True))
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        from da_agent.agent.agents import PromptAgent
        from da_agent.envs import DAAgentEnv

        task_config = copy.deepcopy(self.task.metadata["official_record"])
        task_config["config"] = [
            {
                "type": "copy_all_subfiles",
                "parameters": {"dirs": [str(self.task.source_dir)]},
            }
        ]
        env = DAAgentEnv(
            env_config={
                "init_args": {
                    "name": f"stateguard-{self.task.instance_id}-stage1",
                    "work_dir": "/workspace",
                    "language": language,
                }
            },
            task_config=task_config,
            # ``run.py`` is launched from methods/da-agent and passes
            # ``./cache``. Resolve that exact official cache location.
            cache_dir=str(method_root / "cache"),
            mnt_dir=str(self.workspace.root),
        )
        agent = PromptAgent(
            model=model,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            max_memory_length=max_memory_length,
            max_steps=self.max_steps,
            use_plan=False,
            use_image_prompt=False,
            language=language,
        )
        return agent, env


def _parse_retry_observation(response: str) -> str:
    preview = str(response or "")[:200]
    if re.match(r"^```\s*$", preview.strip()):
        return (
            "Your response contains only code block markers (```). Please provide a "
            "valid action like: Bash(code=\"your command\") or CreateFile(filepath=\"path\")."
        )
    if "Action:" not in preview:
        return (
            "Your response is missing an 'Action:' section. Please format your response "
            "as: Thought: [your reasoning] Action: [valid action]"
        )
    if len(preview.strip()) < 10:
        return "Your response is too short. Please provide a complete thought and action."
    return (
        "Failed to parse action from your response. Please provide a valid action like: "
        'Bash(code="command"), CreateFile(filepath="path"), '
        'EditFile(filepath="path"), LOCAL_DB_SQL(sql_query="query"), or '
        f'Terminate(output="result"). Your response was: {preview}...'
    )


def _public_action_arguments(action: Any) -> dict[str, Any]:
    values = {}
    for name, value in vars(action).items():
        if name.startswith("_"):
            continue
        rendered = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        values[name] = rendered
    return values


def _observation_succeeded(observation: str) -> bool:
    lowered = observation.lower()
    failure_markers = (
        "traceback",
        "syntaxerror",
        "exception:",
        "failed to",
        "no such file",
        "not found",
        "error:",
        "timed out",
    )
    return not any(marker in lowered for marker in failure_markers)
