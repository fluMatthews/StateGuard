import unittest

from stateguard.state.draft import (
    DraftStore,
    RelationFinalization,
    RelationFinalizationMode,
    StateHeader,
)
from stateguard.state.graph import StateRelationGraph
from stateguard.state.models import (
    AnalyticalState,
    Conclusion,
    Constraint,
    StateRelation,
    StateRelationType,
    VariableRef,
)
from stateguard.state.store import StateStore


def make_state(state_id, relation, value):
    variable = VariableRef("metric", state_id, value=value, value_type="int", producer_state_id=state_id)
    return AnalyticalState(
        id=state_id,
        issue=f"Compute metric at {state_id}",
        confidence=0.95,
        constraints=(Constraint("Use the executed value."),),
        used_variables=(variable,),
        conclusions=(Conclusion(f"C{state_id[1:]}", f"metric is {value}", (variable.key,), ("step",)),),
        relations=(relation,),
    )


class StateStoreTest(unittest.TestCase):
    def test_upstream_correction_is_recorded_only_in_current_state(self):
        store = StateStore()
        upstream = make_state("S1", StateRelation(StateRelationType.INIT), 41)
        store.commit(upstream)
        corrected_variable = VariableRef(
            "corrected_metric",
            "S2",
            value=42,
            value_type="int",
            producer_state_id="S2",
        )
        current = AnalyticalState(
            id="S2",
            issue="Use independently verified arithmetic",
            confidence=0.99,
            constraints=(Constraint("Use the independently executed value."),),
            used_variables=(corrected_variable,),
            conclusions=(
                Conclusion(
                    "C2",
                    "The independently verified value is 42.",
                    (corrected_variable.key,),
                    ("isolated-probe",),
                ),
            ),
            relations=(StateRelation(StateRelationType.PROGRESS, "S1"),),
        )
        store.commit(current)

        self.assertEqual(store.get("S1").used_variables[0].value, 41)
        self.assertEqual(store.get("S2").used_variables[0].key, "corrected_metric@S2")
        self.assertEqual(store.get("S2").relations[0].related_state_id, "S1")

    def test_committed_verified_state_is_locked(self):
        store = StateStore()
        state = make_state("S1", StateRelation(StateRelationType.INIT), 1)
        store.commit(state)
        external_copy = store.get("S1")
        external_copy.metadata["tampered"] = True
        self.assertNotIn("tampered", store.get("S1").metadata)
        with self.assertRaises(ValueError):
            store.commit(state)

    def test_variable_version_must_match_current_state_id(self):
        with self.assertRaises(ValueError):
            AnalyticalState(
                id="S2",
                issue="bad version",
                confidence=1.0,
                constraints=(Constraint("use current state version"),),
                used_variables=(VariableRef("metric", "S1", value=2),),
                conclusions=(),
                relations=(StateRelation(StateRelationType.INIT),),
            )

    def test_provisional_relation_can_be_revised_before_commit(self):
        drafts = DraftStore()
        drafts.open(
            StateHeader(
                id="S3",
                issue="Reassess an earlier branch",
                constraints=(Constraint("Use the completed trace."),),
                relations=(StateRelation(StateRelationType.PROGRESS, "S1"),),
            )
        )
        drafts.finalize_relations(
            RelationFinalization(
                mode=RelationFinalizationMode.RESELECT,
                relations=(StateRelation(StateRelationType.BRANCH, "S2"),),
                reason="The completed trace follows S2 rather than the provisional S1 path.",
                conflict_evidence=("The completed state uses S2 output, not S1 output.",),
            )
        )
        state = drafts.close()

        self.assertEqual(state.relations[0].type, StateRelationType.BRANCH)
        self.assertEqual(state.relations[0].related_state_id, "S2")
        self.assertEqual(state.metadata["provisional_relations"][0]["related_state_id"], "S1")
        self.assertEqual(state.metadata["relation_finalization_mode"], "reselect")

    def test_versioned_states_and_relations(self):
        store = StateStore()
        graph = StateRelationGraph()
        s1 = make_state("S1", StateRelation(StateRelationType.INIT), 1)
        s2 = make_state("S2", StateRelation(StateRelationType.PROGRESS, "S1"), 2)
        for state in (s1, s2):
            store.commit(state)
            graph.add_state(state)

        self.assertEqual([item.key for item in store.get("S2").used_variables], ["metric@S2"])
        self.assertEqual(graph.ancestors("S2"), ("S1",))
        self.assertEqual(graph.descendants("S1"), ("S2",))

    def test_invalidate_is_an_ordinary_relation_edge(self):
        graph = StateRelationGraph()
        s1 = make_state("S1", StateRelation(StateRelationType.INIT), 1)
        s2 = make_state("S2", StateRelation(StateRelationType.INVALIDATE, "S1"), 2)
        graph.add_state(s1)
        graph.add_state(s2)
        self.assertEqual(graph.nodes["S1"]["status"], "committed")
        self.assertEqual(graph.edges[-1], {"source": "S1", "target": "S2", "type": "invalidate"})


if __name__ == "__main__":
    unittest.main()
