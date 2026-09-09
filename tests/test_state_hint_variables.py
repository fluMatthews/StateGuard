"""What a Worker cannot cheaply rediscover is worth sending back to it.

Two thirds of the variables a DE state carries are inventories of table or
file names the Worker wrote itself and can list with one shell command. The
rest is what it paid many steps to learn: that a source table holds admin_id
in the team_id column, that an email filter cut 3,004 rows down to 7. Only
that half belongs in the hint, and only where a flow asks for it.
"""

from __future__ import annotations

import unittest

from stateguard.state.models import (
    AnalyticalState,
    Conclusion,
    Constraint,
    StateRelation,
    StateRelationType,
    VariableRef,
    _HINT_VALUE_CHARS,
    _is_inventory,
)


def _state(*variables: VariableRef) -> AnalyticalState:
    return AnalyticalState(
        id="S1",
        issue="Write the staging layer.",
        constraints=(Constraint(text="Pure DuckDB syntax."),),
        used_variables=variables,
        conclusions=(Conclusion(claim="14 staging files were written."),),
        relations=(StateRelation(type=StateRelationType.INIT),),
    )


class InventoryDetectionTests(unittest.TestCase):
    def test_a_list_of_names_is_an_inventory(self):
        self.assertTrue(_is_inventory(["stg_a__x", "stg_a__y", "stg_a__z"]))

    def test_the_same_names_joined_by_commas_are_too(self):
        # Half of them arrive as strings rather than lists.
        self.assertTrue(_is_inventory("stg_a__x, stg_a__y, stg_a__z"))
        self.assertTrue(_is_inventory("a.sql, b.sql, c.sql, d.sql"))

    def test_a_finding_is_not_an_inventory(self):
        # Spaces, parentheses and equals signs are what separate the two.
        self.assertFalse(_is_inventory("admin_id = id (3004 rows; filter keeps 7)"))
        self.assertFalse(
            _is_inventory("contact_id (VARCHAR len 3-50), email (lower(trim(x)) not null)")
        )
        self.assertFalse(_is_inventory({"admin_id": "DB.team_id", "team_id": "DB.admin_id"}))

    def test_a_sentence_followed_by_column_names_survives(self):
        # raw_schema opens on prose and then lists columns; requiring every
        # piece to be bare, not most, is what keeps it.
        self.assertFalse(
            _is_inventory("63 raw tables in raw schema; contact_data has id, email, first_name")
        )

    def test_two_names_are_too_few_to_call_an_inventory(self):
        self.assertFalse(_is_inventory(["only_one", "only_two"]))

    def test_scalars_are_never_inventories(self):
        for value in (True, False, 0, 993, None):
            self.assertFalse(_is_inventory(value))


class StateHintVariableTests(unittest.TestCase):
    def test_variables_are_absent_unless_asked_for(self):
        hint = _state(VariableRef(name="admin_mapping", version="S1", value="a = b (7 rows)")).as_state_hint()
        self.assertEqual(sorted(hint), ["conclusions", "id", "issue", "relations"])

    def test_findings_are_kept_and_inventories_dropped(self):
        state = _state(
            VariableRef(name="team_admin_mapping", version="S1", value={"admin_id": "DB.team_id"}),
            VariableRef(name="marts_files", version="S1", value=["a_x", "b_y", "c_z"]),
            VariableRef(name="dim_user_rows", version="S1", value=993),
        )
        names = [item["name"] for item in state.as_state_hint(include_variables=True)["variables"]]
        self.assertEqual(names, ["team_admin_mapping", "dim_user_rows"])

    def test_a_long_finding_is_truncated_not_dropped(self):
        # The longest real finding runs 466 characters; the cap keeps its head.
        value = "source column order differs from the contract; " + "mapping " * 200
        state = _state(VariableRef(name="source_db_schema", version="S1", value=value))
        kept = state.as_state_hint(include_variables=True)["variables"][0]["value"]
        self.assertTrue(kept.startswith("source column order differs"))
        self.assertLessEqual(len(kept), _HINT_VALUE_CHARS + 1)

    def test_a_boolean_warning_is_kept(self):
        # Nothing but meaning separates job_application_duplication=true from
        # marts_written=true, and four characters is not worth the risk.
        state = _state(VariableRef(name="job_application_duplication", version="S1", value=True))
        self.assertEqual(
            state.as_state_hint(include_variables=True)["variables"],
            [{"name": "job_application_duplication", "value": True}],
        )

    def test_a_state_whose_variables_are_all_inventories_gains_no_field(self):
        state = _state(VariableRef(name="marts", version="S1", value=["a_x", "b_y", "c_z"]))
        self.assertNotIn("variables", state.as_state_hint(include_variables=True))


class FlowScopingTests(unittest.TestCase):
    def test_de_asks_for_variables_and_da_does_not(self):
        from stateguard.adapters.dacomp.workflow import DACompWorkflow

        self.assertTrue(DACompWorkflow(5, hint_includes_variables=True).state_hint_includes_variables())
        self.assertFalse(DACompWorkflow(5).state_hint_includes_variables())

    def test_other_flows_expose_no_such_hook(self):
        # The corpus flow also resumes with state summaries; the getattr in
        # _relation_hint is what keeps its hints unchanged. DABstep's workflow
        # is covered by the suite run in its own environment, where its extra
        # is installed.
        from stateguard.adapters.corpus.workflow import CorpusSingleQueryWorkflow

        self.assertTrue(hasattr(CorpusSingleQueryWorkflow, "resumes_with_state_summary"))
        self.assertFalse(
            hasattr(CorpusSingleQueryWorkflow, "state_hint_includes_variables")
        )


if __name__ == "__main__":
    unittest.main()
