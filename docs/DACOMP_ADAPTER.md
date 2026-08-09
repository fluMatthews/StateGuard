# DAComp adapter

StateGuard integrates the selected DAComp tracks through
`src/stateguard/adapters/dacomp/` without changing the benchmark-independent
harness, state, repair, manager, evidence-tool, or checkpoint code.

## Official protocols retained

### DA stage 1

- Loads the official 100-record `dacomp-da.jsonl` and each instance's SQLite task
  directory. Rubric metadata is evaluator-only and never enters `TaskSpec`, the
  Worker, or the Manager.
- Uses the official `PromptAgent`, `DACOMP_SYSTEM_DESIGN_EN`, action parser,
  `DAAgentEnv`, Bash/SQL/file actions, history truncation, and parse-retry behavior.
- Counts only successfully parsed actions against the official default 120-action
  budget. The adapter does not start the visualization or report-merging agents.
- Exports `<instance_id>.md`, `<instance_id>-traj.txt`, and `result.json` in the
  official evaluator layout.
- The judge bridge calls only the official rubric channel and reports Accuracy and
  Completeness. It does not construct either GSB channel and excludes
  Conclusiveness, Readability, Analytical Depth, and Visualization.

### DE Implementation and Evolution

- Loads exactly the official 30 Impl and 50 Evol directories, requiring the same
  `run.py`, layer config, data contract/question files as the official runner.
- Copies the complete task directory into the prediction directory and calls the
  vendored official `create_de_task_prompt`, language guideline/suffix builders,
  CodeActAgent system prompt/model config, action classes, CLI Runtime, memory/MCP
  setup, and the official 200-action budget from `run_infer_de.sh`.
- OpenHands' stock controller automatically generates the next action as soon as a
  tool observation arrives. `OpenHandsStepwiseControllerSession` subclasses that
  controller and gates only this final auto-step edge; official event handling, runtime
  execution, control flags, malformed-action feedback, stuck detection, memory, and
  native CodeAct messages remain in place. It does not translate the Worker into
  StateGuard's JSON ReAct.
- Runs `python run.py` after the Worker, uses the official trajectory simplifier, and
  writes the official task-directory plus `result.json`/`workspace_summary.json` layout.
- The judge bridge invokes the official deterministic evaluator twice with
  `--force-rebuild`, once in `cs` mode and once in `cfs` mode. Gold directories are
  referenced only by that post-hoc process.

## StateGuard single-query lifecycle

Both tracks literally share `DACompWorkflow`:

1. One DAComp instance creates one fresh Worker, Manager, workspace, state store,
   graph, trace buffer, and checkpoint set.
2. The Worker runs native benchmark actions. After every five accepted Worker
   actions (or a terminal action), the harness pauses. Manager actions never consume
   the Worker budget.
3. Five steps are only a review cadence. The Manager first decides whether pending
   actions contain an important result. If not, it resumes and the pending interval
   continues accumulating.
4. If a state exists, the Manager selects an arbitrary contiguous prefix starting at
   `candidate_start_step`. After a repair, it may select the correct execution-ordered
   subset of the rewritten interval. The endpoint need not be a multiple of five.
5. `OPEN_STATE` has no provisional relations and injects no state hint. Only an
   evidenced repair appends the fixed, non-solving error hint.
6. After writing and checking the state body, the Manager selects relations from the
   compact index and needed state files exactly once with `FINALIZE_RELATIONS/SELECT`.
7. Repair remains the universal per-state schedule: two light hint retries, one heavy
   retry, then restore the original branch and abandon that state. For DAComp's
   file-based workspace, heavy cleanup deletes only an explicitly named
   `file:relative/path`; ambiguous natural-language variables are not destructively
   mapped to files.
8. Checkpoints store source-relative code/text deltas rather than embedding the
   multi-gigabyte SQLite/DuckDB bytes in every snapshot. A rollback restores official
   inputs, reapplies accepted deltas, and rebuilds a derived database only when needed.
   Manager probes lazily create one task-local, read-only data copy and reuse it across
   disposable code scratches; it is deleted when the task closes.

## Commands

Baseline DA stage 1 (no Manager):

```bash
PYTHONPATH=src python -m stateguard.adapters.dacomp.runner \
  --dacomp-root /path/to/DAComp-main \
  --track da-stage1 --model <official-da-model> \
  --output-dir results --task-limit 1
```

DE with a file-handshake Manager:

```bash
PYTHONPATH=src python -m stateguard.adapters.dacomp.runner \
  --dacomp-root /path/to/DAComp-main \
  --track de-evol --model <openhands-llm-config> \
  --manager-file-dir handshakes --output-dir results --task-limit 1
```

Judging is deliberately separate. `--judge` must be explicitly supplied; DA also
requires `--rubrics-model`, while DE can use `--de-eval-python` to point to the
official evaluator environment.

## Environment boundary

The adapter imports benchmark-owned dependencies lazily. DA stage 1 must run in an
environment containing `methods/da-agent/requirements.txt`; DE must run in the vendored
OpenHands Poetry environment. The generic StateGuard unit-test environment is sufficient
for adapter/core tests but is intentionally not treated as a substitute for either
official runtime. No model call or LLM judge is made by import or by `--help`; judging
only occurs with explicit `--judge`.
