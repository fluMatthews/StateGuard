from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Protocol

from stateguard.core.events import ReActStep
from stateguard.core.models import AgentAction, Message, ToolResult



# A DE Worker step is far larger than a DA or DABstep one: it runs a whole
# script through execute_bash, so one step carries the script inline and the
# command's full stdout. Measured on impl-010 a single step reached 185KB while
# the per-step budget is about 9,700 characters (an 84,000-character input
# budget, less the 12,927-character pinned prompt, over the 5-7 steps a DE
# review holds). One such step therefore overflows the window on its own, and
# halving the pending trace cannot help once a single step is all that is left.
#
# ``elide_step`` is the escape hatch the harness reaches for at that point. The
# head carries what the step does (the command's target and opening, the first
# lines of output); the tail carries what it produced (a closing "print(...)" or
# "ls", an exit code, an error) -- both are behaviour, while the elided middle is
# script body and long listings. Command and stdout together land near 8,000
# characters, inside the per-step budget. The Worker's stated intent is never
# touched: it is the densest description of behaviour per byte.
#
# Nothing is elided until the harness asks, and only the step it names. Other
# Workers do not define this method, so their traces are never rewritten.
_ELIDED_MARK = "characters elided"
_COMMAND_HEAD, _COMMAND_TAIL = 2500, 1500
_OUTPUT_HEAD, _OUTPUT_TAIL = 2000, 2000

# The Worker's own chain of thought, recovered from reasoning_content, was the
# one field elision never touched, and it is the largest. Across 763 DE steps
# it runs 856 characters at the median but 38,716 at p99 and 62,220 at most,
# against 20,098 and 24,771 at p99 for command and output. Two reviews died on
# it: one block held 15 steps whose reasoning was 59% of 135,557 characters,
# another held 5 steps whose reasoning was 95% of 141,151 -- both overflowed
# the 40,960-token window before the Manager could answer.
#
# 8,000 is the loosest cap that fixes it. A 20-step block lands at 143,366
# characters at p90 under a 16,000 cap and 135,968 under 12,000, both still
# past the ~120,600 an observation may occupy beside the pinned prompt; 8,000
# brings it to 121,457 and leaves 90.4% of steps untouched. 4,000 saves only
# 18,636 more while eliding 74 more steps.
#
# Head-weighted, in the same 62.5% proportion the command uses. A long DE
# reasoning is a continuous SQL design argument with no conclusion at the end
# -- the sampled 39,842-character one opens on the definition it settled
# ("history_unique_key = id || '|' || created_date || row_number()") and stops
# mid-sentence on a syntax question. The decision it reached is already in the
# command; what only the reasoning holds is why the definition was chosen.
_REASONING_HEAD, _REASONING_TAIL = 5000, 3000


def _elided(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail or _ELIDED_MARK in text:
        return text
    dropped = len(text) - head - tail
    return f"{text[:head]}\n...[{dropped:,} {_ELIDED_MARK}]...\n{text[-tail:]}"


def _elided_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _elided(value, _COMMAND_HEAD, _COMMAND_TAIL)
        if isinstance(value, str)
        else value
        for key, value in arguments.items()
    }


@dataclass(frozen=True)
class NativeCodeActStep:
    action_name: str
    thought: str
    arguments: dict[str, Any]
    observation: str
    execution_attempted: bool
    execution_succeeded: bool
    terminal: bool = False
    final_output: str = ""
    raw_action: str = ""
    raw_observation: str = ""


class DECodeActSession(Protocol):
    def start(self, instruction: str) -> None: ...
    def advance(self) -> NativeCodeActStep: ...
    def inject_user_message(self, content: str) -> None: ...
    def snapshot(self) -> Any: ...
    def restore(self, snapshot: Any) -> None: ...
    def messages(self) -> tuple[Message, ...]: ...
    def trajectory(self) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class DEWorkerSnapshot:
    session: Any
    accepted_steps: int
    done: bool
    final_answer: str | None


class DACompDEWorkerAgent:
    """StateGuard Agent facade over DAComp's official CodeAct action/runtime pair."""

    def __init__(self, session: DECodeActSession, *, max_steps: int = 30) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.session = session
        self.max_steps = max_steps
        self._accepted_steps = 0
        self._done = False
        self._final_answer: str | None = None
        self._started = False

    @property
    def messages(self) -> tuple[Message, ...]:
        return self.session.messages() if self._started else ()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def final_answer(self) -> str | None:
        return self._final_answer

    @property
    def accepted_steps(self) -> int:
        return self._accepted_steps

    def start(self, prompt: Any, system_prompt: str | None = None) -> None:
        if self._started:
            raise RuntimeError("DE CodeAct Worker has already started")
        if system_prompt is not None:
            raise ValueError("DE uses CodeActAgent's official native system prompt")
        self.session.start(str(prompt))
        self._started = True

    def continue_turn(self, prompt: Any) -> None:
        del prompt
        raise RuntimeError("DAComp-DE is single-query")

    def step(self) -> ReActStep:
        if not self._started:
            raise RuntimeError("DE CodeAct Worker has not started")
        if self._done:
            raise RuntimeError("DE CodeAct Worker is already complete")
        if self._accepted_steps >= self.max_steps:
            self._done = True
            raise RuntimeError(
                f"DE CodeAct Worker exhausted the official {self.max_steps}-action budget"
            )
        try:
            native = self.session.advance()
        except Exception as exc:                      # noqa: BLE001
            # A context overflow ends the run rather than propagating: history
            # only grows, so every later step would fail the same way. The steps
            # already taken are kept and written out, matching the DA worker.
            if "contextwindow" not in type(exc).__name__.lower() \
                    and "contextwindow" not in str(exc).lower() \
                    and "context window" not in str(exc).lower():
                raise
            self._done = True
            self._final_answer = ""
            return ReActStep(
                step_id=self._accepted_steps + 1,
                action=AgentAction(
                    kind="final",
                    reasoning="Context window exceeded; no further step is possible.",
                    answer="",
                ),
                observation=None,
                done=True,
                raw_model_output="",
                metadata={
                    "benchmark": "dacomp",
                    "track": "de",
                    "official_stop_reason": "context_length_exceeded",
                    "worker_budget_used": self._accepted_steps,
                    "worker_budget_limit": self.max_steps,
                },
            )
        self._accepted_steps += 1
        budget_exhausted = self._accepted_steps >= self.max_steps
        self._done = native.terminal or budget_exhausted
        if native.terminal:
            self._final_answer = native.final_output
            action = AgentAction(
                kind="final", reasoning=native.thought, answer=native.final_output
            )
            observation = None
        else:
            action = AgentAction(
                kind="tool",
                reasoning=native.thought,
                tool_name=native.action_name,
                arguments=native.arguments,
            )
            observation = ToolResult(
                tool_name=native.action_name,
                ok=native.execution_succeeded,
                output=native.observation,
                data={
                    "execution_attempted": native.execution_attempted,
                    "execution_succeeded": native.execution_succeeded,
                    "native_action": native.raw_action,
                },
                error=None if native.execution_succeeded else native.observation,
            )
            if budget_exhausted:
                self._final_answer = ""
        return ReActStep(
            step_id=self._accepted_steps,
            action=action,
            observation=observation,
            done=self._done,
            raw_model_output=native.raw_action or native.thought,
            metadata={
                "benchmark": "dacomp",
                "track": "de",
                "official_action": native.raw_action,
                "official_observation": native.raw_observation or native.observation,
                "worker_budget_used": self._accepted_steps,
                "worker_budget_limit": self.max_steps,
            },
        )

    def elide_step(self, step: ReActStep) -> ReActStep | None:
        """Shorten one pending step so a review that still overflows can proceed.

        Returns None when the step is already short enough or has been elided
        before, so the harness can tell "cannot shrink further" from "shrunk".
        The rewritten step keeps every field the Manager reads and every field
        the audit needs: ``metadata`` still carries the untouched official action
        and observation, so worker.jsonl remains a complete record.
        """
        action = step.action
        arguments = _elided_arguments(action.arguments or {})
        reasoning = _elided(action.reasoning or "", _REASONING_HEAD, _REASONING_TAIL)
        observation = step.observation
        output = observation.output if observation is not None else ""
        elided_output = _elided(output, _OUTPUT_HEAD, _OUTPUT_TAIL)
        if (
            arguments == (action.arguments or {})
            and reasoning == (action.reasoning or "")
            and elided_output == output
        ):
            return None
        return replace(
            step,
            action=replace(action, arguments=arguments, reasoning=reasoning),
            observation=None
            if observation is None
            else replace(
                observation,
                output=elided_output,
                error=None if observation.error is None else elided_output,
            ),
        )

    def inject_observation(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        del metadata
        self.session.inject_user_message(str(content))
        if self._done and self._accepted_steps < self.max_steps:
            self._done = False
            self._final_answer = None

    def snapshot(self) -> DEWorkerSnapshot:
        return DEWorkerSnapshot(
            self.session.snapshot(),
            self._accepted_steps,
            self._done,
            self._final_answer,
        )

    def restore(self, snapshot: DEWorkerSnapshot) -> None:
        self.session.restore(snapshot.session)
        self._accepted_steps = snapshot.accepted_steps
        self._done = snapshot.done
        self._final_answer = snapshot.final_answer

    def trajectory(self) -> list[dict[str, Any]]:
        return self.session.trajectory()

    def close(self) -> None:
        self.session.close()
