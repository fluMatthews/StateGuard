from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from stateguard.core.models import Message


_PYTHON_RE = re.compile(r"<python>(.*?)</python>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_python(action: str) -> str | None:
    match = _PYTHON_RE.search(action or "")
    if match is None:
        return None
    code = match.group(1)
    fenced = re.search(r"```python(.*?)```", code, re.DOTALL)
    return fenced.group(1) if fenced else code


def extract_answer(action: str) -> str | None:
    match = _ANSWER_RE.search(action or "")
    return match.group(1).strip() if match else None


def reasoning_text(action: str) -> str:
    match = re.search(r"<reasoning>(.*?)</reasoning>", action or "", re.DOTALL)
    return match.group(1).strip() if match else ""


def official_messages(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": str(message["role"]), "content": str(message.get("content", ""))}
        for message in messages
    ]


def core_messages(messages: Iterable[Mapping[str, Any]]) -> tuple[Message, ...]:
    return tuple(
        Message(str(message["role"]), str(message.get("content", "")))
        for message in messages
    )


def wrap_stateguard_observation(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("<information>") and stripped.endswith("</information>"):
        return stripped
    return f"<information>\n{stripped}\n</information>"

