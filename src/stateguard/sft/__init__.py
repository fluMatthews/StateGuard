"""Benchmark-independent Manager SFT construction and context policy."""

from .activation import export_manager_activations
from .context import (
    DEFAULT_MANAGER_CONTEXT_CHARS,
    DEFAULT_MANAGER_OUTPUT_RESERVE_CHARS,
    select_manager_context,
)

__all__ = [
    "DEFAULT_MANAGER_CONTEXT_CHARS",
    "DEFAULT_MANAGER_OUTPUT_RESERVE_CHARS",
    "export_manager_activations",
    "select_manager_context",
]
