# DABstep adapter

## Official protocol retained

The adapter loads the public tasks and seven context files from the downloaded
`adyen/DABstep` dataset and imports the downloaded official baseline factory.
The Worker therefore keeps the official:

- `smolagents==1.3.0` `CodeAgent` / `CustomCodeAgent` split;
- chat and reasoning model prompts;
- `LiteLLMModelWithBackOff` settings and retry policy;
- authorized imports and read-only `open` function;
- persistent local Python interpreter;
- ten code-action steps and the official max-step final-answer fallback; and
- deterministic dev scorer and `answers.jsonl` fields.

StateGuard adds per-task trajectory and state artifacts but does not put them in
the Worker workspace. By default, both manager=None and managed runs use the
version-locked compat-v1 profile: it repairs known smolagents 1.3.0 Python
semantic defects without changing official prompts, authorized imports, model
token settings, or the ten-step Worker budget. Use --runtime-profile official
only in a fresh process for a strict-runtime sanity run. With `manager=None`, no Manager observation, state hint,
error hint, state operation, probe, or checkpoint is executed.

## Managed single-query lifecycle

One DABstep query creates a fresh Worker, Manager, workspace, state store, trace,
and repair runtime. The Manager prompt is initialized before execution, but no
Manager control action is requested until the first three-step pause. The interruptible facade executes one native
`Thought/Code/Observation` action at a time and pauses after every three native
steps or on termination.

The pause is not a state boundary. The Manager first decides whether an important
result has formed. If it has not, the pending trace remains available after the
Worker resumes. If it has, the Manager chooses an interval whose endpoint need
not align with the three-step cadence, writes and verifies the state, and selects
relations only after the body is complete. DABstep never injects a state hint.
Only a justified repair appends a chronological `<manager_feedback>` observation
to the official native memory.

Repair uses the shared two-light/one-heavy schedule. The heavy attempt additionally
removes only named variables from the official persistent Python interpreter.
Spent Worker calls remain spent after rollback, while Manager actions and probes
do not consume the official Worker step budget.

## Outputs

For a run root `OUTPUT/MODEL/SPLIT/EXPERIMENT`:

- `config.yaml` and `answers.jsonl` retain the official aggregate interface;
- `logs.txt` provides an aggregate task summary;
- `tasks/TASK_ID/result.json` and `trajectory.json` preserve the native audit trail;
- `tasks/TASK_ID/stateguard/` contains Manager, repair, trace, state-store, and
  summary sidecars.

The dev reference answer is held only by the dataset/evaluator object. It is not
included in `TaskSpec`, Manager initialization, observations, workspace, state
store, or trajectory artifacts.

## CLI

```bash
PYTHONPATH=src python -m stateguard.adapters.dabstep.runner \
  --dabstep-root /fs/fast/u2024201619/DABstep \
  --output-dir ./runs/dabstep \
  --model-id MODEL_ID \
  --split dev \
  --max-steps 10 \
  --review-cadence 3 \
  --runtime-profile compat-v1
```

Omit all Manager options for the comparable official Worker baseline. Use either
`--manager-model` plus API settings or `--manager-file-dir` for a managed run.
Task-level `--concurrency` matches the official runner's independent-task
parallelism; it never parallelizes Worker and Manager actions inside one query.
