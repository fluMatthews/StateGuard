# Normalized corpus runner

This adapter runs normalized analytical corpora through the shared StateGuard
harness. It does not use benchmark gold during Worker or Manager execution.

Supported layouts:

- `dsbench_v1`: `multi-turn/` and `single-turn/`
- `idabench_v2`: multi-turn task directories containing `task.json` and `data/`

The Worker reuses the same official DSGym/LongDS tagged code-action runtime and
local Jupyter execution service used by the historical LongDS baselines. Use the
existing `stateguard-longds` Python environment; the runner does not require
Docker. Multi-turn tasks reuse one Worker, Manager, kernel, store, and conversation
across all turns. Single-query tasks pause every three Worker action steps by
default and use posthoc state relations without state hints.

Before launching a run, the local DSGym manager must be listening at
`http://localhost:5000` and its configured executors must be available. The runner
performs a read/execute/deallocate preflight and exits before any model request if
the service or the required number of slots is unavailable. With a Manager, budget
two executor slots per concurrent task (one persistent Worker plus at most one
probe); without a Manager, budget one.

Example:

```bash
export PYTHONPATH=/fs/fast/u2024201619/StateGuard/src
export WORKER_API_KEY=...
export WORKER_API_BASE=...
export MANAGER_API_KEY=...
export MANAGER_API_BASE=...

/fs/fast/u2024201619/.envs/stateguard-longds/bin/python \
  -m stateguard.adapters.corpus.runner \
  --source idabench_v2 \
  --corpus-root /fs/fast/u2024201619/IDA-Bench/IDA-Bench-v2 \
  --dsgym-root /fs/fast/u2024201619/DataMind-main/longds/runners/DSGym \
  --output-dir /fs/fast/u2024201619/StateGuard/runs \
  --experiment teacher_pilot \
  --mode multi-turn \
  --task-limit 2 \
  --worker-model deepseek-v4-pro \
  --manager-model deepseek-v4-pro
```

Each task run writes:

- `trajectory.json`: complete public Worker trajectory and per-unit results
- `manager_session.json`: complete raw Manager session
- `stateguard/`: states, store, hints, Worker/Manager/repair event streams
- `sft_data_activations.json`: validated activation-level SFT records
- `sft_export_report.json`: accepted/rejected activation audit
- `private/references.json`: evaluator-only answers and relation provenance

The runner also aggregates accepted records from the tasks launched by that CLI
invocation under `.../exports/<timestamp>/sft_data_activations.json`. Failed tasks
and tasks with no accepted Manager activations are retained for audit but excluded
from the aggregate; the command exits nonzero unless `--allow-partial` is given.
