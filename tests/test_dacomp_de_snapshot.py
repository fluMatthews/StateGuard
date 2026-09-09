"""Snapshot/restore for the OpenHands-backed DE Worker session.

The DE Worker is the only one whose session is a live OpenHands controller.
Its State carries a ``threading.Lock`` and a pickle hook that empties history,
so the generic ``copy.deepcopy`` the other Workers use cannot be applied here.
These tests pin the two properties the harness actually relies on: a snapshot
is independent of the live State, and a restore brings the conversation back.
"""

from __future__ import annotations

import copy
import threading
import unittest
from dataclasses import dataclass, field
from typing import Any

from stateguard.adapters.dacomp.worker_de_controller import (
    ControllerSessionSnapshot,
    OpenHandsStepwiseControllerSession,
)


class FakeConversationStats:
    """Stands in for OpenHands ConversationStats: holds an unpicklable lock."""

    def __init__(self) -> None:
        self._save_lock = threading.Lock()
        self.tokens = 0


@dataclass
class FakeState:
    """Mirrors the State behaviour that broke the real run.

    ``__getstate__`` empties history exactly as OpenHands does, so a test that
    passes here would also pass against the real class for these properties.
    """

    history: list[str] = field(default_factory=list)
    iteration_flag: dict[str, Any] = field(default_factory=dict)
    convo_stats: FakeConversationStats = field(default_factory=FakeConversationStats)
    _view: Any = None
    _history_checksum: Any = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["history"] = []
        state.pop("_view", None)
        state.pop("_history_checksum", None)
        return state


class FakeStateTracker:
    def __init__(self, state):
        self.state = state


class FakeController:
    def __init__(self, state):
        self.state = state
        self.state_tracker = FakeStateTracker(state)
        self._pending_action = object()
        self.delegate = object()
        self._stuck_detector = None

    def get_state(self):
        return self.state_tracker.state


class FakeAgent:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


def _session(state):
    session = object.__new__(OpenHandsStepwiseControllerSession)
    session.controller = FakeController(state)
    session.agent = FakeAgent()
    session._started = True
    session._imports = {"StuckDetector": lambda restored: ("stuck", restored)}
    return session


class DESnapshotTests(unittest.TestCase):
    def setUp(self):
        self.state = FakeState(
            history=["step-1", "step-2"], iteration_flag={"current": 2}
        )
        self.session = _session(self.state)

    def test_deepcopy_is_what_broke_the_real_run(self):
        # Guards the premise: the generic clone the other Workers use fails
        # here, which is why this session needs its own snapshot path.
        with self.assertRaises(TypeError):
            copy.deepcopy(self.state)

    def test_snapshot_keeps_history_and_survives_the_lock(self):
        snapshot = self.session.snapshot()
        self.assertTrue(snapshot.started)
        self.assertEqual(snapshot.state["history"], ["step-1", "step-2"])
        # Runtime statistics are shared by identity, not rewound.
        self.assertIs(snapshot.state["convo_stats"], self.state.convo_stats)
        # Caches are dropped so they rebuild from the restored history.
        self.assertNotIn("_view", snapshot.state)
        self.assertNotIn("_history_checksum", snapshot.state)

    def test_snapshot_is_independent_of_later_worker_steps(self):
        snapshot = self.session.snapshot()
        self.state.history.append("step-3")
        self.state.iteration_flag["current"] = 3
        self.assertEqual(snapshot.state["history"], ["step-1", "step-2"])
        self.assertEqual(snapshot.state["iteration_flag"], {"current": 2})

    def test_restore_brings_the_conversation_back(self):
        snapshot = self.session.snapshot()
        self.state.history.append("step-3")
        self.session.restore(snapshot)
        restored = self.session.controller.get_state()
        self.assertEqual(restored.history, ["step-1", "step-2"])
        self.assertIsInstance(restored, FakeState)
        self.assertIsNone(self.session.controller._pending_action)
        self.assertIsNone(self.session.controller.delegate)
        self.assertEqual(self.session.agent.resets, 1)

    def test_one_snapshot_restores_more_than_once(self):
        # The harness restores a checkpoint and then, on a failed transaction,
        # restores the pre-action state from the same stored fields.
        snapshot = self.session.snapshot()
        self.session.restore(snapshot)
        self.session.controller.get_state().history.append("mutated")
        self.session.restore(snapshot)
        self.assertEqual(
            self.session.controller.get_state().history, ["step-1", "step-2"]
        )

    def test_none_state_round_trips(self):
        session = _session(None)
        session.controller = FakeController(None)
        snapshot = session.snapshot()
        self.assertIsNone(snapshot.state)

    def test_snapshot_shape_is_unchanged(self):
        self.assertIsInstance(self.session.snapshot(), ControllerSessionSnapshot)


if __name__ == "__main__":
    unittest.main()
