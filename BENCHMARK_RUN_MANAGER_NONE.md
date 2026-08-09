# Manager=None benchmark runbook

This runbook covers the three benchmarks that currently have concrete StateGuard
adapters:

1. LongDS / DSGym (68 multi-turn tasks)
2. DAComp: DA Stage 1 (100), DE Implementation (30), and DE Evolution (50)
3. DABstep (10 public dev tasks; 450 default/submission tasks)

The purpose of this run is to generate the **Worker-only baseline** through the
same adapter code that will later be used by StateGuard. Generation and judging
should be separate phases. This document only starts generation unless stated
otherwise.

## 1. Non-negotiable baseline rules

- Run from the StateGuard repository and use the benchmark-specific Python
  environment listed below.
- Do **not** pass `--manager-model`, `--manager-api-*`, or
  `--manager-file-dir`. Omitting all Manager arguments is what selects
  `manager=None`.
- Do not add `--judge` during generation. First verify that all expected Worker
  outputs exist and contain no runtime errors; judge them afterward.
- Keep the Worker model, endpoint, temperature, Worker step budget, runtime
  profile, and selected task IDs unchanged for the later managed arm.
- Never write API keys into this repository, a command file, or a log. Export
  them in the launching shell. Do not commit result directories.
- Use a unique run tag. Reusing a DABstep experiment name or DAComp experiment
  name can overwrite/merge task artifacts.

## 2. Fixed local paths and credentials

Open a fresh shell (preferably a `tmux` window) and define:

```bash
export SG=/fs/fast/u2024201619/StateGurad
export LONGDS_DSGYM=/fs/fast/u2024201619/DataMind-main/longds/runners/DSGym
export LONGDS_DATA=/fs/fast/u2024201619/DataMind-main/longds/dataset/task/longds
export DACOMP=/fs/fast/u2024201619/DAComp-main
export DABSTEP=/fs/fast/u2024201619/DABstep

export RUN_TAG=manager_none_$(date +%Y%m%d_%H%M%S)
export OUT="$SG/runs/benchmark_${RUN_TAG}"
mkdir -p "$OUT/logs"

# Enter these in the shell; never paste a real key into this file.
export WORKER_API_BASE='<OpenAI-compatible base URL>'
read -rsp 'Worker API key: ' WORKER_API_KEY
export WORKER_API_KEY
echo
```

Model names used by the current DeepSeek-v4-pro setup are:

- LongDS: `openai/deepseek-v4-pro`
- DABstep: `openai/deepseek-v4-pro`
- DAComp DA: `deepseek-v4-pro` (a DA model-config name)
- DAComp DE: `deepseek-v4-pro` (an OpenHands LLM-config name)

DAComp does not consume `WORKER_API_BASE`/`WORKER_API_KEY` directly from the
StateGuard CLI:

- DA Stage 1 resolves its model through
  `DAComp-main/methods/da-agent/da_agent/agent/config.py`. The model config must
  exist and normally reads the full chat-completions URL from `API_URL` and the
  token from `AUTH_TOKEN`.
- DE resolves `--model` through the official OpenHands `config.toml` mechanism.
  Confirm that the named LLM config exists and reads credentials from the
  environment. Do not store a token in the tracked TOML file.

The earlier one-task DA/Impl/Evol smoke runs used the name
`deepseek-v4-pro`, but that endpoint registration was runtime/local configuration,
not a portable committed benchmark setting. Verify both DAComp model configs
before starting a full run.

For DA Stage 1, for example:

```bash
export API_URL='<full OpenAI-compatible /chat/completions URL>'
export AUTH_TOKEN="$WORKER_API_KEY"
```

## 3. Preflight (no paid model calls)

### 3.1 Confirm the runner CLIs

```bash
cd "$SG"

PYTHONPATH="$SG/src" \
  /fs/fast/u2024201619/.envs/stateguard-longds/bin/python \
  -m stateguard.adapters.longds.runner --help

PYTHONPATH="$SG/src" \
  conda run -n dacomp-da python \
  -m stateguard.adapters.dacomp.runner --help

PYTHONPATH="$SG/src" \
  conda run -n dacomp-de python \
  -m stateguard.adapters.dacomp.runner --help

PYTHONPATH="$SG/src" \
  /fs/fast/u2024201619/.envs/stateguard-dabstep/bin/python \
  -m stateguard.adapters.dabstep.runner --help
```

### 3.2 LongDS executor service

LongDS requires the local DSGym allocation manager on `127.0.0.1:5000` and its
executor pool (ports 8432-8437 on this machine). Check it before spending API
credit:

```bash
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy="$NO_PROXY"
curl -fsS http://127.0.0.1:5000/status
```

Do not start LongDS if this fails. Follow the local DSGym executor setup rather
than replacing the official environment. There are six executor slots. Use
`--task-concurrency 3` initially; each manager=None task holds one Worker slot.

### 3.3 Recommended one-task smoke tests

Run these before launching the full sets. They make paid calls, so only run them
after the endpoint and model names have been confirmed.

LongDS (one task, first two turns only):

```bash
cd "$SG"
PYTHONPATH="$SG/src" \
  /fs/fast/u2024201619/.envs/stateguard-longds/bin/python \
  -m stateguard.adapters.longds.runner \
  --dsgym-root "$LONGDS_DSGYM" \
  --dataset-path "$LONGDS_DATA" \
  --output-dir "$OUT/smoke_longds" \
  --model openai/deepseek-v4-pro \
  --backend litellm \
  --manager-url http://127.0.0.1:5000 \
  --max-steps 40 \
  --temperature 0 \
  --api-key "$WORKER_API_KEY" \
  --base-url "$WORKER_API_BASE" \
  --task-limit 1 \
  --turn-limit 2
```

DAComp (one task from each selected track):

```bash
cd "$SG"

PYTHONPATH="$SG/src" conda run --no-capture-output -n dacomp-da python \
  -m stateguard.adapters.dacomp.runner \
  --dacomp-root "$DACOMP" \
  --output-dir "$OUT/smoke_dacomp" \
  --track da-stage1 \
  --model deepseek-v4-pro \
  --experiment-name "${RUN_TAG}_smoke" \
  --max-worker-steps 120 \
  --task-limit 1

PYTHONPATH="$SG/src" conda run --no-capture-output -n dacomp-de python \
  -m stateguard.adapters.dacomp.runner \
  --dacomp-root "$DACOMP" \
  --output-dir "$OUT/smoke_dacomp" \
  --track de-impl \
  --model deepseek-v4-pro \
  --experiment-name "${RUN_TAG}_smoke" \
  --max-worker-steps 200 \
  --task-limit 1

PYTHONPATH="$SG/src" conda run --no-capture-output -n dacomp-de python \
  -m stateguard.adapters.dacomp.runner \
  --dacomp-root "$DACOMP" \
  --output-dir "$OUT/smoke_dacomp" \
  --track de-evol \
  --model deepseek-v4-pro \
  --experiment-name "${RUN_TAG}_smoke" \
  --max-worker-steps 200 \
  --task-limit 1
```

The explicit `200` for both DE tracks is mandatory for official-run parity. The
current adapter constructor has a lower fallback when the option is omitted,
whereas the official `run_infer_de.sh` launches DE with 200 iterations.

DABstep (one public dev task):

```bash
cd "$SG"
PYTHONPATH="$SG/src" \
  /fs/fast/u2024201619/.envs/stateguard-dabstep/bin/python \
  -m stateguard.adapters.dabstep.runner \
  --dabstep-root "$DABSTEP" \
  --output-dir "$OUT/smoke_dabstep" \
  --model-id openai/deepseek-v4-pro \
  --experiment "${RUN_TAG}_smoke" \
  --split dev \
  --max-tasks 1 \
  --max-steps 10 \
  --runtime-profile compat-v1 \
  --concurrency 1 \
  --api-base "$WORKER_API_BASE" \
  --api-key "$WORKER_API_KEY"
```

## 4. Full manager=None generation commands

Run each command in its own `tmux` window and redirect stdout/stderr to the
corresponding log. Do not add Manager flags.

### 4.1 LongDS: all 68 tasks

```bash
cd "$SG"
PYTHONPATH="$SG/src" \
  /fs/fast/u2024201619/.envs/stateguard-longds/bin/python \
  -m stateguard.adapters.longds.runner \
  --dsgym-root "$LONGDS_DSGYM" \
  --dataset-path "$LONGDS_DATA" \
  --output-dir "$OUT/longds" \
  --model openai/deepseek-v4-pro \
  --backend litellm \
  --manager-url http://127.0.0.1:5000 \
  --max-steps 40 \
  --temperature 0 \
  --api-key "$WORKER_API_KEY" \
  --base-url "$WORKER_API_BASE" \
  --start-index 0 \
  --task-concurrency 3 \
  2>&1 | tee "$OUT/logs/longds.log"
```

For restartable batches, use disjoint slices such as
`--start-index 0 --task-limit 10`, then `--start-index 10 --task-limit 10`.
Do not overlap slices. One raw task is a single multi-turn context; turns inside
that task must never be split across processes for the real baseline.

### 4.2 DAComp: the three selected tracks

DA Stage 1 (100 tasks, official 120-action budget):

```bash
cd "$SG"
PYTHONPATH="$SG/src" conda run --no-capture-output -n dacomp-da python \
  -m stateguard.adapters.dacomp.runner \
  --dacomp-root "$DACOMP" \
  --output-dir "$OUT/dacomp" \
  --track da-stage1 \
  --model deepseek-v4-pro \
  --experiment-name "$RUN_TAG" \
  --max-worker-steps 120 \
  --start-index 0 \
  2>&1 | tee "$OUT/logs/dacomp_da_stage1.log"
```

DE Implementation (30 tasks, official shell budget 200):

```bash
cd "$SG"
PYTHONPATH="$SG/src" conda run --no-capture-output -n dacomp-de python \
  -m stateguard.adapters.dacomp.runner \
  --dacomp-root "$DACOMP" \
  --output-dir "$OUT/dacomp" \
  --track de-impl \
  --model deepseek-v4-pro \
  --experiment-name "$RUN_TAG" \
  --max-worker-steps 200 \
  --start-index 0 \
  2>&1 | tee "$OUT/logs/dacomp_de_impl.log"
```

DE Evolution (50 tasks, official shell budget 200):

```bash
cd "$SG"
PYTHONPATH="$SG/src" conda run --no-capture-output -n dacomp-de python \
  -m stateguard.adapters.dacomp.runner \
  --dacomp-root "$DACOMP" \
  --output-dir "$OUT/dacomp" \
  --track de-evol \
  --model deepseek-v4-pro \
  --experiment-name "$RUN_TAG" \
  --max-worker-steps 200 \
  --start-index 0 \
  2>&1 | tee "$OUT/logs/dacomp_de_evol.log"
```

The current DAComp runner is sequential within one track. It supports
`--task-limit` and comma-separated `--task-ids` for controlled batches. If
multiple track processes are launched concurrently, they must use disjoint task
sets and the same fixed Worker configuration.

### 4.3 DABstep

For the locally scoreable comparison, run all 10 public dev tasks:

```bash
cd "$SG"
PYTHONPATH="$SG/src" \
  /fs/fast/u2024201619/.envs/stateguard-dabstep/bin/python \
  -m stateguard.adapters.dabstep.runner \
  --dabstep-root "$DABSTEP" \
  --output-dir "$OUT/dabstep" \
  --model-id openai/deepseek-v4-pro \
  --experiment "$RUN_TAG" \
  --split dev \
  --max-tasks -1 \
  --max-steps 10 \
  --runtime-profile compat-v1 \
  --concurrency 4 \
  --api-base "$WORKER_API_BASE" \
  --api-key "$WORKER_API_KEY" \
  2>&1 | tee "$OUT/logs/dabstep_dev.log"
```

For the 450-task submission split, change only:

```text
--split default --max-tasks -1 --experiment "${RUN_TAG}_default"
```

The default split has no public answers, so it produces submissions but cannot
be scored locally. `compat-v1` must be used in both the manager=None and managed
arms; do not compare one arm with `official` and the other with `compat-v1`.

## 5. Output locations and progress checks

### LongDS

Each task has its own official-shaped timestamped directory:

```text
$OUT/longds/longds/<domain>/<dataset>/<task_id>/<model_timestamp>/
  traj.json
  results.json
  results_eval.json   # unevaluated copy until judging
  code.py
  stateguard/
```

Progress/error checks:

```bash
grep -E 'PROGRESS|RESULT|ERROR' "$OUT/logs/longds.log" | tail -n 30
find "$OUT/longds" -name results.json | wc -l
curl -fsS http://127.0.0.1:5000/status
```

Expected full count: 68 `results.json` files. A partial task is still written;
inspect `traj.json` and do not count a directory as successful merely because it
exists.

### DAComp

```text
$OUT/dacomp/<track>/deepseek-v4-pro_<RUN_TAG>/<instance_id>/
  result.json
  run_metadata.json
  ...official track-specific outputs...
```

Progress/error checks:

```bash
grep -E 'PROGRESS|RESULT|ERROR' "$OUT/logs/dacomp_da_stage1.log" | tail -n 30
grep -E 'PROGRESS|RESULT|ERROR' "$OUT/logs/dacomp_de_impl.log" | tail -n 30
grep -E 'PROGRESS|RESULT|ERROR' "$OUT/logs/dacomp_de_evol.log" | tail -n 30
find "$OUT/dacomp" -name run_metadata.json | wc -l
grep -R '"manager_enabled": true' "$OUT/dacomp" --include=run_metadata.json
grep -R '"error": "' "$OUT/dacomp" --include=run_metadata.json
```

Expected counts are 100 DA + 30 Impl + 50 Evol = 180 metadata files. The
`manager_enabled` grep must return nothing. Review every non-null `error` before
judging.

### DABstep

```text
$OUT/dabstep/openai_deepseek-v4-pro/<split>/<RUN_TAG>/
  config.yaml
  answers.jsonl
  tasks/<task_id>/result.json
  tasks/<task_id>/trajectory.json
  tasks/<task_id>/run_metadata.json
```

Progress/error checks:

```bash
grep -E 'PROGRESS|ERROR' "$OUT/logs/dabstep_dev.log" | tail -n 30
wc -l "$OUT/dabstep/openai_deepseek-v4-pro/dev/$RUN_TAG/answers.jsonl"
grep -R '"manager_enabled": true' \
  "$OUT/dabstep/openai_deepseek-v4-pro/dev/$RUN_TAG/tasks" \
  --include=run_metadata.json
grep -R '"error": "' \
  "$OUT/dabstep/openai_deepseek-v4-pro/dev/$RUN_TAG/tasks" \
  --include=run_metadata.json
```

The dev run should have 10 lines in `answers.jsonl`. Dev scoring is the official
deterministic scorer and is written inline; it does not make a judge-model call.

## 6. Completion checklist for handoff

Record the following before reporting a run complete:

- exact git commit plus whether the worktree was dirty;
- run tag and absolute output root;
- Worker model, endpoint identifier (not the secret), temperature, and budgets;
- selected task IDs/ranges and expected/completed/error counts;
- LongDS executor service status and task concurrency;
- DABstep runtime profile (`compat-v1`);
- confirmation that no Manager option was supplied;
- confirmation from DAComp/DABstep metadata that `manager_enabled` is false;
- paths to all generation logs;
- judging status: not started, running, or complete.

Do not compare scores until the manager=None and managed arms use the same-period
endpoint/configuration. Archived DeepSeek outputs have known sampling and endpoint
drift and are not a clean baseline for a new managed run.
