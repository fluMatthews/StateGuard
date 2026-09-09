from __future__ import annotations

from dataclasses import replace

from stateguard.adapters.longds.worker import LongDSWorkerAgent
from stateguard.core.events import ReActStep


class CorpusWorkerAgent(LongDSWorkerAgent):
    """Thin corpus label over the tested DSGym/LongDS code-action Worker."""

    def step(self) -> ReActStep:
        step = super().step()
        return replace(
            step,
            metadata={**step.metadata, "benchmark": "corpus"},
        )
