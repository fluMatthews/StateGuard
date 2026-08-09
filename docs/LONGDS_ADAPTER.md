# LongDS / DSGym adapter

`stateguard.adapters.longds` is the benchmark-owned integration for the official
LongDS DSGym runner. It composes DSGym's backend and `AllocatedCodeEnv`; it does
not route the Worker through StateGuard's generic JSON ReAct agent.

## Compatibility invariants

- One raw LongDS task creates one Worker, Manager, state store, runtime, and code
  environment. A new raw task creates a new set.
- The first turn uses the official system prompt plus user message. Later turns
  append only the official user message.
- The complete Worker conversation and Python container persist across turns.
- DSGym's `<reasoning>`, `<python>`, `<answer>`, and `<information>` protocol is
  preserved. The normalized `ReActStep` is a Manager-only shadow record.
- Worker generations use the official per-turn budget. Manager actions consume a
  separate budget; repair generations still consume the remaining Worker budget.
- State and error hints are appended as user `<information>` observations.
- Ground truth is held in evaluator-only records and is not attached to the
  Manager-visible `TaskSpec`, tools, workspace, or StateGuard artifacts.
- `manager=None` uses the same Worker/backend/environment path without hints,
  Manager probes, state operations, or checkpoint activity.

## Run

Baseline:

```bash
PYTHONPATH=src python -m stateguard.adapters.longds.runner \
  --dsgym-root /path/to/DSGym \
  --dataset-path /path/to/dataset/task/longds \
  --output-dir /path/to/results \
  --model MODEL \
  --backend litellm \
  --manager-url http://localhost:5000 \
  --max-steps 40 \
  --api-key "$WORKER_API_KEY"
```

StateGuard Manager:

```bash
PYTHONPATH=src python -m stateguard.adapters.longds.runner \
  --dsgym-root /path/to/DSGym \
  --dataset-path /path/to/dataset/task/longds \
  --output-dir /path/to/results \
  --model MODEL \
  --backend litellm \
  --manager-url http://localhost:5000 \
  --max-steps 40 \
  --api-key "$WORKER_API_KEY" \
  --manager-model MANAGER_MODEL \
  --manager-api-base "$MANAGER_API_BASE" \
  --manager-api-key "$MANAGER_API_KEY"
```

Add `--judge --judge-api-key ... --judge-base-url ...` to invoke the evaluator
from the official `scripts/longds.py` after the task has ended.

Use `--task-concurrency N` to run up to `N` independent raw tasks concurrently.
The default is `1`. Turns, Worker steps, Manager checks, repairs, workspace, and
state evolution inside one task remain strictly sequential and isolated. With
`--manager-file-dir`, concurrent tasks receive separate subdirectories beneath
that directory. Judging remains outside the task worker threads and still runs
only when `--judge` is explicitly supplied.

### Live external Manager bridge

`--manager-file-dir DIR` is an optional transport adapter for development when a
live Manager process cannot be called through an API. It is not part of the
StateGuard method or lifecycle: it only replaces the Manager model client's
request/response transport. Publish each response atomically after validating
the JSON:

```bash
PYTHONPATH=src python -m stateguard.providers.file_handshake \
  --exchange-dir DIR \
  --sequence N \
  --response-file /path/to/validated-response.json
```

Do not write `completion_N.response.json` directly; exposing a partial file can
otherwise make the runner observe an incomplete response. The reader also waits
briefly for a non-atomic partial write to settle as a defensive fallback.

## Output

Each run follows the official directory shape and writes:

```text
MODEL_TIMESTAMP/
├── bak/turn_N_result.json
├── traj.json
├── results.json
├── results_eval.json
├── code.py
└── stateguard/
    ├── worker.jsonl
    ├── manager.jsonl
    ├── hint.jsonl
    ├── repair.jsonl
    ├── summary.json
    └── state_store/
        ├── store.json
        ├── state_index.json
        └── states/S*.json
```

Without `--judge`, `results_eval.json` is the official-shaped unevaluated copy,
matching the official `LONGDS_NO_JUDGE` path. With `--judge`, it is overwritten by
the official judge output.

## Checkpoint boundary

DSGym does not expose a Python namespace/filesystem snapshot API. The adapter's
current managed checkpoint restores the kernel by restart plus replay of the exact
ordered namespace-operation prefix (Worker cells plus heavy cleanup). Heavy repair
removes only explicitly named Python globals.
This preserves deterministic notebook-style analysis, but arbitrary external side
effects, nondeterministic calls, and files written outside a replayable task
workspace cannot be guaranteed bit-for-bit. A future container snapshot endpoint
can replace this backend without changing the Harness or StateGuard lifecycle.
