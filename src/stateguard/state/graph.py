from __future__ import annotations

import copy
from collections import deque
from typing import Any

from .models import AnalyticalState


class StateRelationGraph:
    """Derived visualization of the relations already stored in each state.

    All five relation types have identical graph mechanics.  In particular,
    ``invalidate`` describes a counterfactual/new branch obtained by changing
    an earlier assumption; it never changes the status of the referenced state.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.variable_producers: dict[str, set[str]] = {}

    def add_state(self, state: AnalyticalState) -> None:
        if state.id in self.nodes:
            raise ValueError(f"graph already contains state: {state.id}")
        for relation in state.relations:
            if relation.related_state_id and relation.related_state_id not in self.nodes:
                raise ValueError(
                    f"graph relation references unknown state: {relation.related_state_id}"
                )
        self.nodes[state.id] = {"issue": state.issue, "status": state.status}
        for relation in state.relations:
            if relation.related_state_id:
                self.edges.append(
                    {
                        "source": relation.related_state_id,
                        "target": state.id,
                        "type": relation.type.value,
                    }
                )
        for variable in state.used_variables:
            self.variable_producers.setdefault(variable.key, set()).add(state.id)

    def ancestors(self, state_id: str) -> tuple[str, ...]:
        reverse: dict[str, list[str]] = {}
        for edge in self.edges:
            reverse.setdefault(edge["target"], []).append(edge["source"])
        return self._walk(state_id, reverse)

    def descendants(self, state_id: str) -> tuple[str, ...]:
        forward: dict[str, list[str]] = {}
        for edge in self.edges:
            forward.setdefault(edge["source"], []).append(edge["target"])
        return self._walk(state_id, forward)

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "nodes": self.nodes,
                "edges": self.edges,
                "variable_producers": self.variable_producers,
            }
        )

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.nodes = copy.deepcopy(snapshot["nodes"])
        self.edges = copy.deepcopy(snapshot["edges"])
        self.variable_producers = copy.deepcopy(snapshot["variable_producers"])

    @staticmethod
    def _walk(start: str, adjacency: dict[str, list[str]]) -> tuple[str, ...]:
        seen: set[str] = set()
        queue = deque(adjacency.get(start, []))
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            queue.extend(adjacency.get(node, []))
        return tuple(sorted(seen))
