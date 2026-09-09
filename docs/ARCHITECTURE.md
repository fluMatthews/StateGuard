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
The manager receives the task context, the pending Worker observation span, its
current draft, the active flow policy, and repair status. Committed states
are deliberately absent from this automatic observation. The manager loads
either compact `state_index.json` or one exact `states/S<ID>.json` only when the
lifecycle requires relation selection or related-state checking. It may inspect
provenance and run disposable probes, then emits one explicit command:

- `OPEN_STATE`, `UPDATE_STATE`, `FINALIZE_RELATIONS`, or `COMMIT_STATE`;
- `RESUME_WORKER`;
- `REPAIR` when a concrete error (or the same error after retry) is established;
- `ABANDON_STATE` after the checked heavy retry still has a clear problem;
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

It then traces worker steps into a mutable draft and writes state-ID-versioned variables
and conclusions. Only after that complete draft is visible does it
validate the provisional relation against the actual current state and selected
relation-state contents. It confirms the relation unless concrete evidence shows
a clear conflict. Only then may it reselect using the current state and all stored
states. State content cannot change after relation finalization.

```text
id
issue
constraints: natural-language text plus optional checking code
used_variables: name@state_id=value (value may be omitted)
conclusions: plain natural-language claims
relations: init | progress | branch | invalidate | combine
```

Relation cardinality is semantic: a state with no upstream uses `init`; a state
with exactly one upstream uses `progress`, `branch`, or `invalidate`; a state
with two or more upstream states uses one `combine` edge for each distinct
upstream state. Multiple state IDs are never encoded into one relation field.

Step span, checkpoint reference, status, and metadata are provenance fields, not
substitutes for the six method fields.

Relation search is deliberately absent. `StateStore` persists one aggregate JSON
list at `state_store/store.json` and one immutable artifact per committed state at
`state_store/states/S<ID>.json`, plus a compact `state_store/state_index.json`
containing only id, issue, and conclusions. The harness supplies this compact
index automatically at lifecycle-defined relation stages; the manager calls
`load_state` only when checking one exact related ID. It writes exact IDs itself; related-state checking
is limited to direct, one-hop upstream IDs and cannot recurse through those
parents or traverse downstream. The graph is only a derived end-of-task
visualization/debug artifact. No lexical, BM25, or embedding search is part of
the core method.

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
(`corrected_name@current_state_id`). This requires explicit, independently verified evidence; the upstream state, its historical trace and variables,
and the relation remain untouched; ambiguity always passes without correction.

## One pipeline, two flow adapters

The paper-level method and agent skeleton stay unified. A flow adapter changes
only three execution semantics:

| Semantic point | `TurnFlowAdapter` | `FixedStepFlowAdapter(5)` |
|---|---|---|
| Boundary/state formation | One state per turn; harness binds the whole turn automatically | Review every five steps/final; manager writes one inclusive source interval with arbitrary observed start/end, independent of review cadence |
| Relation timing | Query-first provisional IDs; confirm unless explicit conflict requires reselection | No header relation; select once after the complete current state is written |
| State hint | Inject contents selected by provisional relation IDs | Same hook, with an empty selected-ID set |

In turn flow, an outer benchmark adapter supplies successive turn queries. At
the start of each turn, the harness supplies the current query and compact
committed-state index, and the manager opens the state with provisional relation
IDs. The harness then injects the selected state hint and starts the Worker
mechanically. After UPDATE_STATE, the harness supplies the compact index again;
the manager calls `load_state` only for exact states needed to confirm or, on
explicit conflict, reselect the provisional relation. Repeating this naturally produces the task-level
relation graph.

In fixed-step flow, the worker runs five uninterrupted ReAct steps. At the review
point the manager may simply resume, leaving the pending interval intact, or may
open a state based on that output. Its initial header contains only the state ID,
query constraints, and an empty relation set. The issue, compact variables, plain conclusions, and coarse source interval are
written only after the manager identifies the state. The interval may start and
end at any observed pending steps, so neither boundary must align with the review
cadence. Earlier omitted steps are passed; later steps remain pending. The span is
observation context, not a claim that every enclosed step is correct evidence.
After repair, the same state preserves its start and may extend its end to cover
the retry outcome. The header contains no relation. Once the selected current state
exists, the harness supplies the compact entries together with the current state,
and the manager selects final relation IDs exactly once before commit.

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
    └─ ABANDON_STATE                     (after failed heavy: restore and pass)
```

`ACTION_RESULT` observations let the same long-lived manager continue after a
state operation without hiding policy inside the harness. After every repair,
the manager sees the next worker step and decides whether the same error remains.
The manager itself is deliberately excluded from worker-branch checkpoints, so
it retains the prior evidence and hint needed for this comparison. Rejected
attempt steps remain in an append-only trace ledger and are not treated as
authoritative evidence merely because they fall inside a state's coarse source span.
Light and heavy both preserve conversation and append the same structured
error hint. Heavy has exactly one extra operation: deleting only the explicitly
localized erroneous variables. After every retry, all non-header draft content is cleared and the manager must rewrite it using the original analysis and retry outcome before commit. Multi-turn source remains the complete turn. Single-query source preserves its start and may extend its end through the retry; the Manager never enumerates individual steps. This does not rewrite context or delete artifacts. If two light repairs and one heavy
repair all fail, the manager emits `ABANDON_STATE`; the executor restores the
first pre-repair worker branch and passes (do-no-harm/fail-open).

The light/light/heavy counter belongs to the current state, not the task, turn,
or query. Every newly opened state starts a fresh independent three-attempt budget.

The same logical manager session is retained throughout one single-query task or
one multi-turn task. Its system prompt and controller/task initialization remain
pinned. Only the model-facing tail is bounded: old complete manager action blocks
are dropped first, while an observation, its tool calls/results, and terminal
decision are retained or removed together. The full session remains available in
run artifacts. Worker steps are persisted once in `worker.jsonl`; the in-memory
`TraceBuffer` is only the transactional index that marks pending, drafted,
accepted, rejected, or passed steps and is not a second trace artifact.

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
