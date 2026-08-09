from __future__ import annotations

import importlib.util
from pathlib import Path


def load_official_prompts(dsgym_root: Path) -> tuple[str, str]:
    """Load DSGym's SYSTEM_PROMPT and JUDGE_PROMPT from its source of truth."""
    prompt_path = dsgym_root.expanduser().resolve() / "scripts" / "prompt.py"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"official DSGym prompt.py not found: {prompt_path}")
    spec = importlib.util.spec_from_file_location("stateguard_longds_official_prompt", prompt_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official prompt module: {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.SYSTEM_PROMPT), str(module.JUDGE_PROMPT)


def render_turn_messages(
    *,
    system_prompt_template: str,
    data_root: Path,
    context: str,
    question: str,
    first_turn: bool,
) -> tuple[dict[str, str], ...]:
    user_content = f"{context}\nQuestion: {question}"
    if first_turn:
        return (
            {
                "role": "system",
                "content": system_prompt_template.format(PATH=str(data_root)),
            },
            {"role": "user", "content": user_content},
        )
    return ({"role": "user", "content": user_content},)

