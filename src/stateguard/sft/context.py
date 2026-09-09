from __future__ import annotations

from collections.abc import Sequence

from stateguard.core.models import Message

# Budgeted in characters, spent against a token window. Measured on real
# Manager sessions the ratio runs 3.12-4.28 chars per token, so 100_000 is
# at worst ~32k tokens and clears the 40_960 window with room for the ratio
# to move. The previous 120_000 was chosen for offline SFT export, where no
# window applies, and it produced 41k-46k token requests the server refused.
DEFAULT_MANAGER_CONTEXT_CHARS = 100_000
DEFAULT_MANAGER_OUTPUT_RESERVE_CHARS = 16_000


def select_manager_context(
    messages: Sequence[Message],
    max_context_chars: int | None = DEFAULT_MANAGER_CONTEXT_CHARS,
    reserved_output_chars: int = DEFAULT_MANAGER_OUTPUT_RESERVE_CHARS,
) -> list[Message]:
    """Select complete Manager blocks for runtime inference or SFT export.

    The system message and controller/task initialization are pinned. The
    newest block is always retained, then older complete blocks are added
    newest-first while the input budget permits. The full logical session is
    never mutated.
    """
    selected_messages = list(messages)
    if max_context_chars is None or len(selected_messages) <= 2:
        return selected_messages
    if max_context_chars < 1:
        raise ValueError("max_context_chars must be positive or None")
    if reserved_output_chars < 0:
        raise ValueError("reserved_output_chars cannot be negative")

    pinned = selected_messages[:2]
    blocks = _manager_blocks(selected_messages[2:])
    if not blocks:
        return pinned

    input_budget = max(1, max_context_chars - reserved_output_chars)
    newest_index = len(blocks) - 1
    unit_init_indices = [
        index
        for index, block in enumerate(blocks)
        if block
        and block[0].content.lstrip().startswith("<task_unit_initialization>")
    ]
    required_indices = {newest_index}
    if unit_init_indices:
        required_indices.add(unit_init_indices[-1])
    required_blocks = [
        blocks[index] for index in sorted(required_indices)
    ]

    used = sum(
        _message_size(message)
        for message in pinned
        + [message for block in required_blocks for message in block]
    )
    retained_history: list[tuple[int, list[Message]]] = []
    for index in range(newest_index - 1, -1, -1):
        if index in required_indices:
            continue
        block = blocks[index]
        block_size = sum(_message_size(message) for message in block)
        if used + block_size > input_budget:
            break
        retained_history.append((index, block))
        used += block_size

    selected_blocks = [
        block
        for _, block in sorted(
            retained_history
            + [(index, blocks[index]) for index in required_indices],
            key=lambda item: item[0],
        )
    ]
    return pinned + [
        message for block in selected_blocks for message in block
    ]


def _manager_blocks(messages: Sequence[Message]) -> list[list[Message]]:
    blocks: list[list[Message]] = []
    current: list[Message] = []
    for message in messages:
        if message.metadata.get("manager_block_start") and current:
            blocks.append(current)
            current = []
        current.append(message)
    if current:
        blocks.append(current)
    return blocks


def _message_size(message: Message) -> int:
    """Measure a message the way the replay digest reads it.

    Run-local staged paths are normalized out of the replay digest, so two runs
    of the same trajectory hash identically no matter where their run directory
    sits. Budgeting by the raw text broke that: a branch directory whose name is
    a few characters longer than its parent's shifted this running total, which
    dropped a different number of historical blocks and changed the digest of an
    otherwise identical prefix. Measuring the normalized text keeps the budget
    decision independent of the run directory, exactly as the digest already is.
    """
    # Imported here because ``stateguard.counterfactual`` imports this module.
    from stateguard.counterfactual.hashing import normalize_runtime_paths

    return len(normalize_runtime_paths(message.content)) + len(message.name or "")
