from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stateguard.core.models import Message
from stateguard.runtime.workspace import safe_clone

from .worker_de import NativeCodeActStep
from .workspace import DACompWorkspace


@dataclass(frozen=True)
class ControllerSessionSnapshot:
    state: Any
    started: bool


def _step_reasoning(action: Any) -> str:
    """Return the Worker's stated intent for one step, read-only.

    OpenHands fills ``action.thought`` from the assistant message content. A
    reasoning model that answers with a tool call leaves that content empty and
    puts its text in ``reasoning_content``, which the tool-calling branch of
    ``response_to_actions`` never reads -- so every DE step reached the Manager
    with an empty reasoning while DA and DABstep steps carry theirs.

    The whole response is already attached to the action as tool-call metadata,
    so recovering the text here costs the Worker nothing: ``action.thought``
    stays exactly as OpenHands set it, and that field alone is what feeds the
    Worker's own conversation history.
    """
    thought = str(getattr(action, "thought", "") or "")
    if thought:
        return thought
    response = getattr(getattr(action, "tool_call_metadata", None), "model_response", None)
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    if not isinstance(message, dict):
        message = getattr(message, "__dict__", {})
    return str(message.get("reasoning_content") or "").strip()


class OpenHandsStepwiseControllerSession:
    """Official OpenHands controller with one scheduling edge deliberately gated.

    The stock controller calls the next LLM step immediately from its observation
    callback. This subclass keeps all official event handling, runtime execution,
    memory recall, control flags, stuck detection, malformed-action feedback, and
    CodeAct prompt construction, but turns that one callback edge into a condition
    variable. ``advance`` releases exactly one official controller iteration.
    """

    def __init__(
        self,
        *,
        official_root: Path,
        workspace: DACompWorkspace,
        llm_config_name: str,
        max_steps: int = 30,
        agent_class: str = "CodeActAgent",
        language: str = "en",
        wait_timeout: float = 900.0,
    ) -> None:
        self.official_root = official_root.expanduser().resolve(strict=True)
        self.workspace = workspace
        self.llm_config_name = llm_config_name
        self.max_steps = max_steps
        self.agent_class = agent_class
        self.language = language
        self.wait_timeout = wait_timeout
        self._imports: dict[str, Any] = {}
        self._condition = threading.Condition()
        self._started = False
        self._initial_ready = False
        self._advancing = False
        self._captured_action: Any | None = None
        self._captured_observation: Any | None = None
        self._cycle_ready = False
        self.agent: Any | None = None
        self.runtime: Any | None = None
        self.memory: Any | None = None
        self.controller: Any | None = None

    def start(self, instruction: str) -> None:
        if self._started:
            raise RuntimeError("OpenHands controller session already started")
        self._load_official()
        I = self._imports
        # Resolve the official config.toml from the benchmark root instead of the
        # process working directory, which the official CLI happens to rely on.
        config_toml = self.official_root / "methods" / "de-agent" / "config.toml"
        llm_config = I["get_llm_config_arg"](self.llm_config_name, str(config_toml))
        if llm_config is None:
            raise ValueError(
                f"unknown official OpenHands LLM config {self.llm_config_name!r} in {config_toml}"
            )
        # Credentials may be supplied through the environment so they never have to
        # be written into the benchmark's tracked config file.
        api_key = os.environ.get("WORKER_API_KEY")
        if api_key:
            # LLMConfig stores the key as a pydantic SecretStr.
            secret_type = type(llm_config.api_key)
            llm_config.api_key = (
                secret_type(api_key)
                if hasattr(llm_config.api_key, "get_secret_value")
                else api_key
            )
        api_base = os.environ.get("WORKER_API_BASE")
        if api_base:
            llm_config.base_url = api_base
        sandbox = I["get_default_sandbox_config_for_eval"]()
        config = I["OpenHandsConfig"](
            default_agent=self.agent_class,
            run_as_openhands=False,
            runtime="cli",
            max_iterations=self.max_steps,
            sandbox=sandbox,
            workspace_base=str(self.workspace.root),
            enable_browser=False,
        )
        config.set_llm_config(llm_config)
        agent_config = config.get_agent_config(self.agent_class)
        # On a context overflow OpenHands would emit a CondensationRequestAction
        # and carry on with a compressed history, which is itself a form of state
        # management. Measuring what an external state manager adds requires a
        # baseline that does not manage its own memory, so the overflow is
        # surfaced instead and the run stops with the trajectory it has, matching
        # how the DA worker behaves.
        agent_config.enable_history_truncation = False
        agent_config.enable_prompt_extensions = False
        agent_config.enable_jupyter = False
        agent_config.enable_browsing = False

        # Match process_instance + run_controller's construction order: the
        # CLI Runtime is created first with its own fresh event-stream session;
        # the CodeAct registry/agent and memory are then created for the run.
        runtime_registry = I["LLMRegistry"](config)
        self.runtime = I["create_runtime"](config, runtime_registry)
        I["call_async_from_sync"](self.runtime.connect)
        # Official initialize_runtime performs this direct check before memory
        # and controller construction. It never enters model-visible history.
        self.runtime.run_action(I["CmdRunAction"](command="ls -la ."))

        sid = I["generate_sid"](config)
        registry, convo_stats, config = I["create_registry_and_convo_stats"](
            config, sid, None
        )
        self.agent = I["create_agent"](config, registry)
        self.memory = I["create_memory"](
            runtime=self.runtime,
            event_stream=self.runtime.event_stream,
            sid=sid,
            selected_repository=None,
            repo_directory=None,
            conversation_instructions=None,
            working_dir=config.workspace_mount_path_in_sandbox,
        )
        if self.agent.config.enable_mcp:
            _, servers = I["OpenHandsMCPConfigImpl"].create_default_mcp_server_config(
                config.mcp_host, config, None
            )
            self.runtime.config.mcp.stdio_servers.extend(servers)
            I["call_async_from_sync"](
                I["add_mcp_tools_to_agent"],
                120.0,
                self.agent,
                self.runtime,
                self.memory,
            )
        owner = self
        BaseController = I["AgentController"]
        Action = I["Action"]
        Observation = I["Observation"]

        class PausableAgentController(BaseController):
            async def _on_event(inner, event: Any) -> None:
                if getattr(event, "hidden", False):
                    return
                inner.state_tracker.add_history(event)
                if isinstance(event, Action):
                    await inner._handle_action(event)
                elif isinstance(event, Observation):
                    await inner._handle_observation(event)
                owner._observe_controller_event(event)
                # Deliberately omit only BaseController's final automatic
                # ``await inner._step_with_exception_handling()`` edge.

        self.controller = PausableAgentController(
            agent=self.agent,
            event_stream=self.runtime.event_stream,
            convo_stats=convo_stats,
            iteration_delta=config.max_iterations,
            budget_per_task_delta=config.max_budget_per_task,
            agent_to_llm_config=config.get_agent_to_llm_config_map(),
            headless_mode=True,
            confirmation_mode=config.security.confirmation_mode,
        )
        self.runtime.event_stream.add_event(
            I["MessageAction"](content=instruction), I["EventSource"].USER
        )
        self._started = True
        # Memory recall is asynchronous. It is safe to proceed after the pending
        # recall action clears; an empty-memory implementation may produce no event.
        self._wait_initial_ready()

    def advance(self) -> NativeCodeActStep:
        if not self._started or self.controller is None:
            raise RuntimeError("OpenHands controller session has not started")
        I = self._imports
        current_state = self.controller.get_agent_state()
        if current_state in {
            I["AgentState"].FINISHED,
            I["AgentState"].REJECTED,
            I["AgentState"].ERROR,
            I["AgentState"].PAUSED,
            I["AgentState"].STOPPED,
        }:
            return NativeCodeActStep(
                "controller_terminal",
                "",
                {},
                str(getattr(self.controller.get_state(), "last_error", "") or ""),
                False,
                False,
                terminal=True,
            )
        with self._condition:
            self._advancing = True
            self._captured_action = None
            self._captured_observation = None
            self._cycle_ready = False
        if self.controller.get_agent_state() != I["AgentState"].RUNNING:
            I["call_async_from_sync"](
                self.controller.set_agent_state_to, 15.0, I["AgentState"].RUNNING
            )
        I["call_async_from_sync"](self.controller._step_with_exception_handling)
        deadline = time.monotonic() + self.wait_timeout
        with self._condition:
            while not self._cycle_ready:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._advancing = False
                    raise TimeoutError("timed out waiting for one official CodeAct iteration")
                self._condition.wait(min(remaining, 1.0))
            action = self._captured_action
            observation = self._captured_observation
            self._advancing = False

        if action is None:
            content = str(getattr(observation, "content", "") or observation or "")
            state = self.controller.get_agent_state()
            terminal = state in {
                I["AgentState"].FINISHED,
                I["AgentState"].REJECTED,
                I["AgentState"].ERROR,
                I["AgentState"].PAUSED,
                I["AgentState"].STOPPED,
            }
            return NativeCodeActStep(
                "controller_error",
                "",
                {},
                content,
                False,
                False,
                terminal=terminal,
                raw_action="",
                raw_observation=str(observation or content),
            )
        thought = _step_reasoning(action)
        raw_action = str(action)
        if isinstance(action, I["AgentFinishAction"]):
            outputs = getattr(action, "outputs", {}) or {}
            final = outputs if isinstance(outputs, str) else json.dumps(
                outputs, ensure_ascii=False, default=str
            )
            return NativeCodeActStep(
                "finish", thought, {}, "", False, True, True, final, raw_action, ""
            )
        if isinstance(action, I["MessageAction"]):
            content = str(getattr(action, "content", "") or "")
            # Official run_controller calls the fake user only when the agent
            # actually requests a response. Non-waiting messages need no reply.
            if bool(getattr(action, "wait_for_response", False)):
                self.inject_user_message(
                    I["make_codeact_user_response"](self.language)(
                        self.controller.get_state()
                    )
                )
            return NativeCodeActStep(
                "message", thought, {"content": content}, "auto-continue", False, True,
                raw_action=raw_action,
            )
        output = str(getattr(observation, "content", "") or "")
        return NativeCodeActStep(
            _action_name(action),
            thought,
            _action_arguments(action),
            output,
            True,
            _observation_succeeded(observation),
            raw_action=raw_action,
            raw_observation=str(observation or output),
        )

    def inject_user_message(self, content: str) -> None:
        if not self._started or self.runtime is None:
            raise RuntimeError("OpenHands controller session has not started")
        with self._condition:
            self._initial_ready = False
        self.runtime.event_stream.add_event(
            self._imports["MessageAction"](content=content),
            self._imports["EventSource"].USER,
        )
        self._wait_initial_ready()

    # OpenHands keeps live runtime handles on State. ``convo_stats`` owns a
    # ``threading.Lock``, which deepcopy refuses, and it carries token/cost
    # statistics rather than analytical state, so a rollback must not rewind it.
    _SHARED_STATE_FIELDS = ("convo_stats",)
    # Caches State rebuilds from history; ``State.__getstate__`` drops them too.
    _DERIVED_STATE_FIELDS = ("_view", "_history_checksum")

    @classmethod
    def _clone_fields(cls, fields: dict[str, Any] | None) -> dict[str, Any] | None:
        if fields is None:
            return None
        return {
            name: value if name in cls._SHARED_STATE_FIELDS else safe_clone(value)
            for name, value in fields.items()
        }

    @classmethod
    def _state_fields(cls, state: Any) -> dict[str, Any] | None:
        """Snapshot OpenHands State without going through its pickle hooks.

        ``copy.deepcopy`` routes through ``State.__getstate__``, which empties
        ``history`` because OpenHands rebuilds it from the event stream -- a
        replay a StateGuard rollback never performs, so the restored Worker
        would silently lose its conversation. That hook also drags in
        ``convo_stats`` and its lock, which deepcopy cannot handle at all.
        Reading the instance dict directly keeps the conversation, shares the
        runtime statistics by identity, and drops the caches so they rebuild
        from the restored history.
        """
        if state is None:
            return None
        return cls._clone_fields(
            {
                name: value
                for name, value in vars(state).items()
                if name not in cls._DERIVED_STATE_FIELDS
            }
        )

    def snapshot(self) -> ControllerSessionSnapshot:
        state = self.controller.get_state() if self.controller is not None else None
        return ControllerSessionSnapshot(self._state_fields(state), self._started)

    def restore(self, snapshot: ControllerSessionSnapshot) -> None:
        if self.controller is None:
            raise RuntimeError("cannot restore an uninitialized OpenHands controller")
        # Clone on the way out as well: one snapshot can be restored more than
        # once -- the harness restores a checkpoint, then restores the
        # pre-action state when a transaction fails -- so the stored fields
        # must never be handed out by reference.
        restored = object.__new__(type(self.controller.get_state()))
        vars(restored).update(self._clone_fields(snapshot.state) or {})
        self.controller.state_tracker.state = restored
        self.controller.state = restored
        self.controller._pending_action = None
        self.controller.delegate = None
        self.controller._stuck_detector = self._imports["StuckDetector"](restored)
        self.agent.reset()
        self._started = snapshot.started

    def messages(self) -> tuple[Message, ...]:
        if self.controller is None:
            return ()
        rendered: list[Message] = []
        for event in self.controller.get_state().history:
            source_obj = getattr(event, "source", "")
            source = str(getattr(source_obj, "value", source_obj)).lower()
            role = "user" if source in {"user", "environment"} else "assistant"
            rendered.append(Message(role, str(getattr(event, "content", "") or event)))
        return tuple(rendered)

    def trajectory(self) -> list[dict[str, Any]]:
        if self.controller is None:
            return []
        history = self.controller.get_state().history
        pairs = self._imports["compatibility_for_eval_history_pairs"](history)
        return self._imports["simplify_histories"](pairs)

    def close(self) -> None:
        if self.controller is not None:
            try:
                self._imports["call_async_from_sync"](
                    self.controller.close, set_stop_state=False
                )
            except Exception:
                pass
            self.controller = None
        if self.runtime is not None:
            for name in ("shutdown", "close", "stop", "terminate"):
                method = getattr(self.runtime, name, None)
                if not callable(method):
                    continue
                try:
                    if inspect.iscoroutinefunction(method):
                        self._imports["call_async_from_sync"](method)
                    else:
                        method()
                    break
                except Exception:
                    continue
            try:
                self.runtime.event_stream.close()
            except Exception:
                pass
            self.runtime = None

    def _observe_controller_event(self, event: Any) -> None:
        I = self._imports
        with self._condition:
            if not self._advancing:
                if isinstance(event, I["Observation"]):
                    pending = getattr(self.controller, "_pending_action", None)
                    if pending is None:
                        self._initial_ready = True
                        self._condition.notify_all()
                return
            if isinstance(event, I["Action"]):
                source_obj = getattr(event, "source", "")
                source = str(getattr(source_obj, "value", source_obj)).lower()
                if source == "agent" and not isinstance(event, I["SystemMessageAction"]):
                    self._captured_action = event
                    if isinstance(event, (I["AgentFinishAction"], I["MessageAction"])):
                        self._cycle_ready = True
                        self._condition.notify_all()
                return
            if isinstance(event, I["Observation"]):
                if self._captured_action is None:
                    terminal_state = (
                        isinstance(event, I["AgentStateChangedObservation"])
                        and event.agent_state
                        in {
                            I["AgentState"].FINISHED,
                            I["AgentState"].REJECTED,
                            I["AgentState"].ERROR,
                            I["AgentState"].PAUSED,
                            I["AgentState"].STOPPED,
                        }
                    )
                    if isinstance(event, I["ErrorObservation"]) or terminal_state:
                        self._captured_observation = event
                        self._cycle_ready = True
                        self._condition.notify_all()
                    return
                cause = getattr(event, "cause", None)
                action_id = getattr(self._captured_action, "id", None)
                if cause == action_id:
                    self._captured_observation = event
                    self._cycle_ready = True
                    self._condition.notify_all()

    def _wait_initial_ready(self) -> None:
        deadline = time.monotonic() + min(self.wait_timeout, 60.0)
        with self._condition:
            while not self._initial_ready:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for OpenHands memory recall")
                self._condition.wait(min(remaining, 0.5))

    def _load_official(self) -> None:
        method_root = self.official_root / "methods" / "de-agent"
        root_string = str(method_root.resolve(strict=True))
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        modules = {
            "config": importlib.import_module("openhands.core.config"),
            "setup": importlib.import_module("openhands.core.setup"),
            "schema": importlib.import_module("openhands.core.schema"),
            "controller": importlib.import_module("openhands.controller.agent_controller"),
            "stuck": importlib.import_module("openhands.controller.stuck"),
            "events": importlib.import_module("openhands.events"),
            "actions": importlib.import_module("openhands.events.action"),
            "observations": importlib.import_module("openhands.events.observation"),
            "llm_registry": importlib.import_module("openhands.llm.llm_registry"),
            "serialization": importlib.import_module("openhands.events.serialization.event"),
            "async_utils": importlib.import_module("openhands.utils.async_utils"),
            "utils": importlib.import_module("openhands.utils.utils"),
            "shared": importlib.import_module("evaluation.utils.shared"),
            "mcp_config": importlib.import_module("openhands.core.config.mcp_config"),
            "mcp": importlib.import_module("openhands.mcp"),
        }
        actions = modules["actions"]
        observations = modules["observations"]
        self._imports = {
            "OpenHandsConfig": modules["config"].OpenHandsConfig,
            "get_llm_config_arg": modules["config"].get_llm_config_arg,
            "create_runtime": modules["setup"].create_runtime,
            "create_agent": modules["setup"].create_agent,
            "create_memory": modules["setup"].create_memory,
            "generate_sid": modules["setup"].generate_sid,
            "create_registry_and_convo_stats": modules["utils"].create_registry_and_convo_stats,
            "LLMRegistry": modules["llm_registry"].LLMRegistry,
            "AgentState": modules["schema"].AgentState,
            "AgentController": modules["controller"].AgentController,
            "StuckDetector": modules["stuck"].StuckDetector,
            "EventSource": modules["events"].EventSource,
            "Action": actions.Action,
            "MessageAction": actions.MessageAction,
            "SystemMessageAction": actions.SystemMessageAction,
            "AgentFinishAction": actions.AgentFinishAction,
            "CmdRunAction": actions.CmdRunAction,
            "Observation": observations.Observation,
            "ErrorObservation": observations.ErrorObservation,
            "AgentStateChangedObservation": observations.AgentStateChangedObservation,
            "event_to_dict": modules["serialization"].event_to_dict,
            "call_async_from_sync": modules["async_utils"].call_async_from_sync,
            "OpenHandsMCPConfigImpl": modules["mcp_config"].OpenHandsMCPConfigImpl,
            "add_mcp_tools_to_agent": modules["mcp"].add_mcp_tools_to_agent,
            "get_default_sandbox_config_for_eval": (
                modules["shared"].get_default_sandbox_config_for_eval
            ),
            "compatibility_for_eval_history_pairs": (
                modules["shared"].compatibility_for_eval_history_pairs
            ),
            "simplify_histories": importlib.import_module(
                "evaluation.benchmarks.dacomp.run_infer_de"
            ).simplify_histories,
            "make_codeact_user_response": importlib.import_module(
                "evaluation.benchmarks.dacomp.run_infer_de"
            ).make_codeact_user_response,
        }


def _action_name(action: Any) -> str:
    return {
        "CmdRunAction": "execute_bash",
        "IPythonRunCellAction": "execute_ipython_cell",
    }.get(type(action).__name__, type(action).__name__)


def _action_arguments(action: Any) -> dict[str, Any]:
    return {
        key: getattr(action, key)
        for key in ("command", "code", "path", "content")
        if hasattr(action, key)
    }


def _observation_succeeded(observation: Any) -> bool:
    if observation is None or "error" in type(observation).__name__.lower():
        return False
    extras = getattr(observation, "extras", {}) or {}
    metadata = extras.get("metadata", {}) if isinstance(extras, dict) else {}
    exit_code = metadata.get("exit_code") if isinstance(metadata, dict) else None
    if exit_code is not None:
        return int(exit_code) == 0
    return True
