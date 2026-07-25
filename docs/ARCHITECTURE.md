# StateGuard agent-pipeline architecture

## Actual implementation goal

Build the smallest benchmark-independent execution core that realizes the
StateGuard method as **manager-driven, harness-executed** control. The worker
remains a standard interruptible ReAct agent. A manager ReAct agent observes the
task at start and the worker only at adapter-defined review points, then autonomously
chooses whether to open/update/commit analytical state, inspect evidence,
resume the worker, localize an error, or initiate repair.

This package implements the agent pipeline only. Dataset adapters, SFT/RL data
construction, training, and benchmark evaluation are intentionally separate.

## Method-to-code mapping

| Method component | Implementation |
|---|---|
| Standard worker ReAct | `agents/react.py`, `agents/worker.py` |
| Persistent autonomous state manager | `agents/manager.py`, `prompts/manager_controller.txt` |
| Turn/fixed-step execution semantics | `adapters/flow.py` |
| Manager-owned state lifecycle | `state/draft.py`, `validation/models.py` |
| Full analytical state | `state/models.py` |
| Exact-ID five-relation state records | `state/models.py`, `state/store.py` |
| Derived graph visualization | `state/graph.py` |
| Versioned authoritative store | `state/store.py` |
| Optional executable/code checks | `validation/pipeline.py`, `runtime/evidence_tools.py` |
| Evidence-grounded localization | `validation/models.py`, manager controller prompt |
| Adapter-selected state hint injection | `harness/engine.py`, `adapters/flow.py` |
| Fixed two-light/one-heavy repair schedule | `repair/controller.py` |
| Append-only branch ledger | `runtime/trace.py`, `harness/engine.py` |
| Task-bound runtime/checkpoints | `runtime/bundle.py`, `runtime/checkpoints.py` |
| Manager probe workspace | `runtime/executors.py`, `runtime/evidence_tools.py` |
| Worker Python execution and task files | `agents/worker.py`, `runtime/executors.py`, `runtime/workspace.py` |
| GT information firewall | `harness/blind_view.py` |
| Reproducibility artifacts | `telemetry/artifacts.py` |

## Decision and authority boundary

The worker may mutate only its task workspace through worker tools. It runs
uninterrupted until the adapter boundary (turn end or five steps), then pauses.
The manager receives the query, the latest step, untraced steps, its current
draft, the committed-state index, the content of all stored states, exact selected
relation states, and a workspace manifest. It may inspect provenance and run disposable
probes, then emits one explicit command:

- `OPEN_STATE`, `UPDATE_STATE`, `FINALIZE_RELATIONS`, or `COMMIT_STATE`;
- `RESUME_WORKER`;
- `REPAIR` when a concrete error (or the same error after retry) is established;
- `ROLLBACK_PASS` after the checked heavy retry still has a clear problem;
- `ABSTAIN`.

These decisions belong to the manager. The harness never detects a boundary,
searches for a related state, runs a mandatory check, or judges correctness.
Once the manager triggers `REPAIR`, the method-level executor follows the fixed
two-light/one-heavy schedule; this phase transition is not inferred by harness.

Only `StateGuardHarness` can:

- validate and execute a manager command;
- persist a manager-authored analytical state;
- add state-relation edges;
- restore the original branch only after repair exhaustion;
- remove explicitly identified erroneous workspace variables on heavy repair;
- inject a state hint or error hint into the worker;
- choose the authoritative branch.

The harness owns these side effects for transactional safety, not their semantic
policy. Safety limits such as maximum actions/repairs are likewise mechanical.

## Analytical state

In turn flow, the manager first writes an immutable header before tracing the new state:

```text
id
issue
constraints
relations: provisional exact IDs selected from the query and stored-state contents
```

It then traces worker steps into a mutable draft and writes state-ID-versioned variables,
conclusions, and confidence. Only after that complete draft is visible does it
validate the provisional relation against the actual current state and selected
relation-state contents. It confirms the relation unless concrete evidence shows
a clear conflict. Only then may it reselect using the current state and all stored
states. State content cannot change after relation finalization.

```text
id
issue
confidence
constraints
used_variables: variable_name@state_id (for example, cleaned_df@S4)
conclusions
relations: init | progress | branch | invalidate | combine
```

Step span, checkpoint reference, status, and metadata are provenance fields, not
substitutes for the seven method fields.

Relation search is deliberately absent. The manager sees both a compact index and
the actual stored-state issues, variable values/versions, conclusions, and relations.
It reasons over that content and writes exact IDs itself; later loading and
runtime checking may read only the current state's direct, one-hop upstream
relation IDs from `StateStore`; it cannot recurse through those parents or traverse
downstream. The graph is
only a derived end-of-task visualization/debug artifact. No lexical, BM25, or embedding search
is part of the core method.

`invalidate` is only one of the five relation labels. It represents a new
counterfactual branch formed by changing a prior state's assumption or condition.
It does not invalidate, deactivate, delete, or mutate the referenced state.

Once committed, a state is verified and locked. `StateStore` rejects ID
overwrites and returns defensive copies. The manager can repair and rewrite only
the current open draft; direct related states are read-only evidence.

The manager does not normally audit those related verified states. In the rare
case that a direct upstream value is unmistakably wrong under explicit
constraints and independent execution/probe evidence, it may compute a corrected
value and record it only as a new variable owned by the current state
(`corrected_name@current_state_id`). This requires exceptional confidence
(target at least 0.95). The upstream state, its historical trace and variables,
and the relation remain untouched; ambiguity always passes without correction.

## One pipeline, two flow adapters

The paper-level method and agent skeleton stay unified. A flow adapter changes
only three execution semantics:

| Semantic point | `TurnFlowAdapter` | `FixedStepFlowAdapter(5)` |
|---|---|---|
| Boundary/state formation | One state per turn; review at turn end | Review every five steps/final; manager decides whether the interval forms state |
| Relation timing | Query-first provisional IDs; confirm unless explicit conflict requires reselection | No header relation; select once after the complete current state is written |
| State hint | Inject contents selected by provisional relation IDs | Same hook, with an empty selected-ID set |

In turn flow, an outer benchmark adapter supplies successive turn queries. At
the start of each turn, the manager reads the current query and full committed
state contents, then opens the state with provisional relation IDs. After tracing
and checking the turn it compares the actual state with those selected states. It
confirms by default; if and only if an explicit conflict exists, it uses the same
current-state/full-store selection procedure as single-query. Repeating this
naturally produces the task-level relation graph.

In fixed-step flow, the worker runs five uninterrupted ReAct steps. At the review
point the manager may simply resume, or may open a state based on that output,
attach the selected trace, and check it. The header contains no relation. Once the
complete current state exists, the manager reads it together with all stored-state
contents and selects final relation IDs exactly once before commit.

`TaskAdapter` remains responsible for turning one raw benchmark task into one or
more `TaskSpec` units, creating one task-level workspace, and submission formatting.
The caller reuses one harness across a LongDS task's turn units and creates a new
harness for the next independent task. `FlowAdapter` is deliberately separate:
it controls StateGuard timing, relation phase, and state-hint policy. Both flows
use the same action schema and the same `StateGuardHarness` implementation.

## Autonomous control and transaction protocol

```text
TASK_START
    ↓
turn: manager OPEN_STATE(header + provisional relation IDs) before trace
fixed-step: manager initially RESUME_WORKER
    ↓
worker runs uninterrupted to the adapter boundary → worker pauses
    ↓
manager independently chooses:
    ├─ RESUME_WORKER                    (no important state action yet)
    ├─ UPDATE_STATE                     (attach selected trace to open draft)
    ├─ FINALIZE_RELATIONS               (turn: confirm/reselect; fixed: select once)
    ├─ COMMIT_STATE                     (persist state + graph transactionally)
    ├─ inspect/probe, then act again     (manager tool use)
    ├─ REPAIR                            (next fixed light/light/heavy attempt)
    └─ ROLLBACK_PASS                     (after failed heavy: restore and pass)
```

`ACTION_RESULT` observations let the same long-lived manager continue after a
state operation without hiding policy inside the harness. After every repair,
the manager sees the next worker step and decides whether the same error remains.
The manager itself is deliberately excluded from worker-branch checkpoints, so
it retains the prior evidence and hint needed for this comparison. Rejected
attempt steps remain in an append-only trace ledger but never enter committed
state. Light and heavy both preserve conversation and append the same structured
error hint. Heavy has exactly one extra operation: deleting only the explicitly
localized erroneous variables. After every retry, all non-header draft content
is cleared and the manager must rewrite it from the new worker trace before commit; it
does not rewrite context or delete artifacts. If two light repairs and one heavy
repair all fail, the manager emits `ROLLBACK_PASS`; the executor restores the
first pre-repair worker branch and passes (do-no-harm/fail-open).

The light/light/heavy counter belongs to the current state, not the task, turn,
or query. Every newly opened state starts a fresh independent three-attempt budget.

`StateGuardRuntime` binds worker, manager, workspace, store, draft, trace ledger,
graph, artifacts, and composite checkpoints by object identity. A mismatched
checkpoint component is rejected at construction. Manager evidence tools are
wired to this same store/workspace by default. Manager API, parsing,
illegal-action, and tool-chain failures are recorded as `ManagerFailure`; the
failed action is reverted and the worker resumes fail-open, with the run summary
marked `degraded`.

## Baseline and official-runtime parity

`StateGuardHarness(manager=None)` is a true bypass: it calls the worker until its
native terminal condition and never builds a manager view, injects a hint, writes
state, or creates a repair branch. Thus baseline and StateGuard runs can share the
same worker object, tools, prompt, executor, and benchmark task adapter.

`TurnFlowAdapter` models DSGym's essential lifecycle: subsequent turn queries are
appended to the existing transcript, the workspace persists, and only the per-turn
completion flag and step count reset. `FixedStepFlowAdapter` models an independent
single-query ReAct task and starts a fresh transcript. Exact benchmark integration
should wrap the official DSGym or Smolagents worker behind the same `Agent` protocol,
keeping its native prompt, tools, stopping rule, and transcript format unchanged;
StateGuard adds only the adapter pause hooks and observation injection.

## Production extension points

- Implement `ModelClient` for another inference backend.
- Register task and manager tools through separate `ToolRegistry` objects.
- Replace `InMemoryWorkspace` and `TrustedPythonExecutor` with notebook,
  container, SQL, or remote execution adapters.
- Supply a composite runtime-specific checkpoint manager.
- Implement the thin `TaskAdapter` interface for a benchmark.

`TrustedPythonExecutor` is intentionally labeled for trusted local tests; a
production data-agent run should use an isolated process or container backend.
