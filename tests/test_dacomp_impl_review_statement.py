"""DE-Impl's statement embeds the whole data_contract.yaml, so the Manager's
pinned prompt was 165,716 characters -- 94% of a 40,960-token window, against
0.2% for the observation. The Manager cannot budget a pinned message away, so
the track that has the problem supplies a shortened statement for review.

Only DE-Impl does. DE-Evol carries 12,423 to 39,438 characters of requirements
written for that task alone, at most 28% of the window, and is passed through
whole -- as is every other benchmark, none of which sets the key at all.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from stateguard.adapters.dacomp.dataset import (
    DACompTask,
    DACompTrack,
    _contract_digest,
    _impl_review_statement,
    _render_contract,
)
from stateguard.agents.manager import StateManagerAgent
from stateguard.core.models import TaskSpec


class FakeClient:
    def complete(self, messages, tools):  # pragma: no cover - never called
        raise NotImplementedError


def _task(track: DACompTrack, instruction: str) -> DACompTask:
    return DACompTask(
        instance_id="dacomp-de-x-001",
        track=track,
        instruction=instruction,
        source_dir=Path("/tmp"),
        metadata={},
    )


def _manager_query(spec: TaskSpec) -> str:
    agent = StateManagerAgent(FakeClient())
    agent.configure_lifecycle("SINGLE-QUERY LIFECYCLE")
    agent.start_task(spec)
    prompt = agent.messages[-1].content
    return json.loads(prompt.split("\n\nTASK:\n", 1)[1])["query"]


class ImplReviewStatementTests(unittest.TestCase):
    def setUp(self):
        # The contract opens at character 2,656 in all 30 tasks and runs to the
        # last byte, so every instruction is in the head.
        self.head = ("## Task Description:\n" + "requirement " * 400)[:2_656]
        self.contract = "```yaml\n" + "  - name: column\n" * 9_000 + "```\n"

    def test_the_contract_body_is_elided_and_the_instructions_kept(self):
        reviewed = _impl_review_statement(self.head + self.contract)
        self.assertIn("## Task Description:", reviewed)
        self.assertIn(self.head[-50:], reviewed)
        self.assertIn("characters of data contract elided", reviewed)
        self.assertLess(len(reviewed), 4_200)

    def test_a_statement_that_already_fits_is_returned_verbatim(self):
        for length in (200, 2_142, 4_000):
            text = "x" * length
            self.assertEqual(_impl_review_statement(text), text)

    def test_only_de_impl_supplies_a_review_statement(self):
        long_text = self.head + self.contract
        impl = _task(DACompTrack.DE_IMPL, long_text).task_spec()
        evol = _task(DACompTrack.DE_EVOL, long_text).task_spec()
        self.assertIn("manager_query", impl.metadata)
        self.assertNotIn("manager_query", evol.metadata)

    def test_the_worker_statement_is_never_shortened(self):
        spec = _task(DACompTrack.DE_IMPL, self.head + self.contract).task_spec()
        self.assertEqual(spec.query, self.head + self.contract)
        self.assertIn("- name: column", spec.query)

    def test_the_manager_reviews_against_the_shortened_de_impl_statement(self):
        spec = _task(DACompTrack.DE_IMPL, self.head + self.contract).task_spec()
        self.assertLess(len(_manager_query(spec)), 4_200)

    def test_the_manager_reviews_against_the_whole_de_evol_statement(self):
        # 39,438 characters is DE-Evol's longest; it fits and is all requirements.
        instruction = "## Specific Business Requirements\n" + "field constraint " * 2_300
        spec = _task(DACompTrack.DE_EVOL, instruction).task_spec()
        self.assertEqual(_manager_query(spec), instruction)

    def test_a_task_without_the_key_reviews_against_its_own_query(self):
        # LongDS, DABstep and DAComp DA all take this path.
        spec = TaskSpec(id="t", query="Which issuing country leads?", metadata={})
        self.assertEqual(_manager_query(spec), "Which issuing country leads?")




class ContractDigestTests(unittest.TestCase):
    """The cut statement named no table and no rule, so the Manager repeated
    the four formatting requirements as its constraints and, on one task,
    pushed the Worker to cast eight staging tables the contract never asked to
    be cast. The digest carries the contract's executable part instead.
    """

    CONTRACT = """
staging_contract:
  tables:
    - name: stg_x__lead
      description: Cleaned leads
      source_table: raw.x_lead
      grain: One row per lead
      columns:
        - name: lead_id
          data_type: VARCHAR
          constraints: [not_null, unique]
        - name: email
          data_type: VARCHAR
          validation_rules:
            - rule: "email = LOWER(TRIM(email))"
              on_failure: correct
            - rule: "email ~ '^[a-z]+@[a-z]+$'"
              on_failure: delete_row
        - name: notes
          data_type: VARCHAR
      row_filters:
        - rule: "_deleted IS NULL OR _deleted = false"
          on_failure: delete_row
modeling_spec:
  intermediate_models:
    - name: int_x__enriched
      grain: One row per lead
      source_models:
        - staging.stg_x__lead
        - staging.stg_x__account
      business_logic: 'Start from staging.stg_x__lead and derive things.'
      columns:
        - name: lead_id
          data_type: VARCHAR
          source_expression: "id"
"""

    def digest(self):
        return _contract_digest(self.CONTRACT)

    def test_lineage_and_grain_are_kept(self):
        text = self.digest()
        self.assertIn("stg_x__lead <- raw.x_lead | One row per lead", text)
        self.assertIn(
            "int_x__enriched <- staging.stg_x__lead, staging.stg_x__account | One row per lead",
            text,
        )

    def test_a_columns_rules_are_kept_with_their_outcome(self):
        text = self.digest()
        self.assertIn("email: email = LOWER(TRIM(email)) -> correct", text)
        self.assertIn("email: email ~ '^[a-z]+@[a-z]+$' -> delete_row", text)

    def test_a_rules_own_quotes_survive(self):
        # Only a wrapping quote belongs to YAML; the regex delimiters are content.
        self.assertIn("'^[a-z]+@[a-z]+$'", self.digest())

    def test_source_expression_is_kept(self):
        self.assertIn("lead_id = id", self.digest())

    def test_constraints_are_kept_when_a_column_has_no_rule(self):
        self.assertIn("lead_id [not_null, unique]", self.digest())

    def test_row_filters_are_kept(self):
        self.assertIn("row filter: _deleted IS NULL OR _deleted = false -> delete_row", self.digest())

    def test_a_column_with_no_instruction_is_absent(self):
        # notes only has a name and a type; listing every such column would cost
        # 22% of the contract and invite casts the contract never asked for.
        self.assertNotIn("notes", self.digest())

    def test_prose_stays_out(self):
        text = self.digest()
        self.assertNotIn("Cleaned leads", text)
        self.assertNotIn("business_logic", text)
        self.assertNotIn("derive things", text)

    def test_a_declared_type_other_than_varchar_is_named(self):
        # gold honours data_type for 10,076 of the 10,386 columns that carry no
        # source_expression, 97.0%. Omitting it left five TIMESTAMP columns as
        # text on one task and cost 9.4 points.
        contract = self.CONTRACT.replace(
            "        - name: notes\n          data_type: VARCHAR\n",
            "        - name: created_date\n          data_type: TIMESTAMP\n"
            "        - name: amount\n          data_type: DOUBLE\n"
            "        - name: opened_at\n          data_type: TIMESTAMP\n",
        )
        text = _contract_digest(contract)
        self.assertIn("TIMESTAMP: created_date, opened_at", text)
        self.assertIn("DOUBLE: amount", text)

    def test_varchar_columns_are_not_named(self):
        # VARCHAR is what a passthrough already is; naming every one of them
        # costs 3,100 characters a task at the median for nothing.
        self.assertNotIn("VARCHAR", self.digest())

    def test_a_contract_that_yaml_cannot_parse_still_yields_a_digest(self):
        # dacomp-de-impl-020 holds a regex whose escape makes safe_load raise.
        import yaml

        broken = self.CONTRACT.replace(
            '              on_failure: correct\n',
            '              on_failure: correct\n        - name: bad\n          rule: "x ~ \'^[a-z]+\\.[a-z]+$\'"\n',
        )
        try:
            yaml.safe_load(broken)
        except yaml.YAMLError:
            pass
        self.assertIn("stg_x__lead <- raw.x_lead", _contract_digest(broken))

    def test_the_statement_carries_the_digest_and_keeps_the_task_description(self):
        head = ("## Task Description:\n" + "requirement " * 400)[:2_656]
        statement = head + "```yaml\n" + self.CONTRACT + "  - name: pad\n" * 9_000 + "```\n"
        reviewed = _impl_review_statement(statement)
        self.assertIn("## Task Description:", reviewed)
        self.assertIn("stg_x__lead <- raw.x_lead", reviewed)
        self.assertIn("email: email = LOWER(TRIM(email)) -> correct", reviewed)

    def test_models_are_dropped_whole_once_the_budget_is_spent(self):
        tables = [
            {"name": f"m{i}", "sources": ["raw.s"], "grain": "", "columns": [], "filters": []}
            for i in range(40)
        ]
        rendered = _render_contract(tables, 200)
        self.assertIn("more models omitted", rendered)
        self.assertLessEqual(len(rendered), 260)


if __name__ == "__main__":
    unittest.main()
