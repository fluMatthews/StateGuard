# StateGuard

StateGuard is a benchmark-independent agent pipeline for maintaining evolving,
verifiable analytical states and repairing cascading errors in long-horizon
data-agent tasks.

The package implements one minimal end-to-end vertical slice:

1. a role-neutral ReAct core;
2. an almost-unmodified worker and a long-lived, tool-using manager controller;
3. manager-owned state boundaries, checking, localization, and repair decisions;
4. relation selection from the current query/trace and a compact state index;
5. the full analytical-state schema and five state-relation types from the method;
6. evidence-grounded validation and structured error hints;
7. a fixed two-light/one-heavy repair protocol with append-only attempt tracing;
8. blind manager views, on-demand exact state loading, and append-only run artifacts.

Benchmark loading, datasets, benchmark-specific SFT file I/O, RL, and leaderboard
evaluators remain outside this core package. Generic Manager activation splitting,
validation, and runtime-aligned context selection live in `src/stateguard/sft`.

## Architecture

```text
                         ┌──── exact state IDs / checks / commands ────┐
                         │                                              │
Worker ReAct ──flow segment──> StateGuardHarness ──observation──> Manager ReAct
      ▲                         │   │                              │
      └──resume / hint──────────┘   └──execute command────────────┘
 task tools + workspace        state / graph / checkpoint       read/probe tools
```

The manager is the semantic controller: it decides state formation where the
active flow permits it, which earlier state IDs a state relates to, the coarse
source interval in single-query flow, whether and how to check it, and whether
to commit, repair, or pass. Multi-turn source binding remains harness-owned. The harness is a mechanism-only
scheduler and action executor. It pauses/wakes agents, validates commands,
persists manager-authored state, restores the original branch after exhausted
repair (and uses transactional checkpoints), and injects observations;
it does not infer boundaries, relevance, correctness, or whether repair is
warranted. Once triggered, repair follows the method's fixed light/light/heavy
schedule. Light appends a hint without rewriting context; heavy additionally
removes only explicit erroneous variables. Checkpoint restoration occurs only
after all three attempts fail.

Light and heavy always inject the same structured error hint. Heavy's only extra
operation is exact erroneous-variable deletion. A retry clears the current
draft's non-header content, so the manager must rewrite it from the new trace.
Each state has its own independent two-light/one-heavy budget.

Variables use their producing analytical state ID as the version identifier
(`variable_name@S4`), not an independent integer counter. `invalidate` is a plain
relation label for a counterfactual branch and never deactivates an old state.
Stored relation IDs are the runtime source of truth; the graph is derived only
for visualization/debugging. Committed states are verified and immutable to the
manager. Current-state checking may inspect only direct upstream relation states
(one hop), which remain read-only.

Committed states are persisted as an aggregate JSON list in
`state_store/store.json` and as individual immutable files in
`state_store/states/S<ID>.json`. They are not automatically repeated in every
manager observation: the manager explicitly loads the complete store for
relation selection or one exact state for direct related-state checking.

An apparent upstream-value error is corrected only under exceptionally strong,
independently executed evidence. The replacement is written as a new variable in
the current state; the verified upstream state and relation are never modified.
Uncertain cases pass without correction.

One pipeline supports two execution flows through `FlowAdapter`:

- `TurnFlowAdapter`: one state per turn; the harness binds the whole turn and the
  manager never selects steps. Relation IDs are chosen from the current query,
  then confirmed unless clear conflict evidence requires reselection; related
  state hints are injected.
- `FixedStepFlowAdapter(5)`: review every five Worker steps; the manager decides
  whether pending analysis forms a state and, if so, writes one inclusive
  `source_interval={start,end}`. The interval may start and end at any observed
  pending steps and need not align with review cadence. After repair its start is
  preserved and its end may extend. Relations are selected posthoc; no state hint
  is injected.

The related-state hint sent to the worker contains only each selected state's
`id`, `issue`, conclusions, and relations. Full variables,
constraints, and provenance remain manager-side evidence and are not injected.

The adapters change timing and observation policy only. Worker/manager agents,
actions, harness execution, state store/graph, validation, and repair are shared.

One logical manager session spans an entire single-query task or multi-turn task.
Its initial system/controller messages are pinned; later context is truncated
only in complete observation/action blocks so tool calls never lose their
results. The same block selector is reused by `stateguard.sft` when converting a
complete Manager session into activation-level training records. Worker execution
is persisted once in `worker.jsonl`; the in-memory trace ledger stores only
transactional step status used by the harness.

Passing `manager=None` activates the same worker runtime without any StateGuard
observation, hint, state write, checkpoint, or repair. This is the decoupled
baseline path. `TurnFlowAdapter` preserves the complete worker conversation and
workspace across successive calls while resetting only the per-turn ReAct step
budget; `FixedStepFlowAdapter` starts a fresh independent query. An official
benchmark worker can implement the small `Agent` runtime protocol and keep its
native prompt, tools, executor, stopping rule, and transcript format unchanged.

`WorkerAgent` is bound to the runtime's persistent `worker_executor` and receives
a standard `python(code=...)` FunctionTool in both baseline and StateGuard modes.
Benchmark adapters pass real files through `TaskSpec.data_files`; before the
first ReAct step, the harness stages/registers them in the same workspace and
exposes their paths through the Python namespace's `data_files` mapping.

## Run the tests

The core has no third-party runtime dependency:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Minimal example

```bash
PYTHONPATH=src python examples/minimal_pipeline.py
```

The directory is named `StateGuard` to match the requested workspace path; the
importable Python package is correctly named `stateguard`.
