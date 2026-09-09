# Counterfactual Worker replay

This module adds an offline branch path without changing any normal Worker or
Manager runner. A completed Worker run remains the replay source of truth.

## Semantics

1. Extract an immutable call manifest from a successful Worker run.
2. Start a fresh official execution workspace.
3. Replay recorded Worker and Manager model responses without API calls while
   executing their actions normally in the fresh workspace.
4. Verify every replayed model input against the parent manifest.
5. Replace exactly one selected Worker response. The shared replay switch is
   activated by that Worker call and the original recorded suffix is never consumed.
6. By default, subsequent Manager and Worker calls are live. Optionally supply a
   complete Manager Tool/control sequence ending in REPAIR. Those responses are
   returned before the live Manager, while the normal ReAct engine and harness still
   execute every tool, lifecycle transition, and REPAIR.
7. Optionally supply verified Worker repair responses. Each response is eligible
   only after the harness has accepted and applied a Manager REPAIR. Once optional
   demonstrations are consumed, both agents continue live.

The injection coordinate is `(unit_id, worker_step_id)`. Manager events are
recorded for prefix reproduction and audit but are not mutation points. Manager
failures, protocol retries, failed tools, or missing states never disqualify an
otherwise complete Worker trajectory from intervention.

## Intervention file

```json
{
  "intervention_id": "example-001",
  "target_unit_id": "dsbench_v1/example/turn_2",
  "target_step_id": 3,
  "replacement_action": "<reasoning>...</reasoning><python>...</python>",
  "forced_manager_responses": [],
  "forced_repair_responses": [],
  "metadata": {
    "operator": "defined-later"
  }
}
```

`replacement_action` is the full Worker action returned to the official
environment. It may alter reasoning, code, or both. When present,
`forced_manager_responses` must contain valid Manager Tool/control actions and end
with exactly one REPAIR. Each response uses peek/ack/reject semantics: a successful
Tool result or lifecycle-accepted control acknowledges and removes it; rejection or
execution failure stops the demonstration and invalidates the branch instead of
advancing to the next response. After the final REPAIR is applied, execution returns
to the live Manager. `forced_repair_responses` contains Worker actions and is consumed
only after the harness accepts and applies a Manager REPAIR. Leaving either optional
list empty preserves the corresponding live-agent behavior.

Replay eligibility and sample selection are separate. Intervention requires every
Worker unit to complete with an answer and without Worker budget exhaustion; it does
not require clean Manager behavior or SFT eligibility. Manager issues are retained in
manifest metadata for post-hoc filtering. Parent artifacts are hashed and the
StateGuard Python/prompt tree is fingerprinted. A later corpus-specific semantic and
quality selector decides which clean branches become training triples. Runtime timestamps and isolated
`worker_data` paths are normalized; all other input drift aborts replay.

## Entrypoints

Manifest-only extraction:

```bash
python -m stateguard.counterfactual.manifest \
  --run-dir CLEAN_RUN \
  --output CLEAN_RUN/replay_manifest.json
```

Batch-prepare all eligible Worker runs without model calls:

```bash
python -m stateguard.counterfactual.prepare \
  --runs-root CLEAN_RUNS_ROOT \
  --report MANIFEST_REPORT.json
```

No-API reproducibility audit:

```bash
python -m stateguard.adapters.corpus.counterfactual_runner \
  --replay-only \
  --parent-run CLEAN_RUN \
  --corpus-root CORPUS_ROOT \
  --dsgym-root DSGYM_ROOT \
  --output-dir OUTPUT_ROOT
```

Counterfactual continuation uses the same command with `--intervention`, Worker
API options, and Manager API options. It writes an isolated branch directory
containing the normal corpus artifacts plus `replay_manifest.json`,
`intervention.json`, and `replay_report.json`.
