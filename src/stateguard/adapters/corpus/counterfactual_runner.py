from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from stateguard.agents.react import parse_action, parse_json_object
from stateguard.agents.manager import StateManagerAgent
from stateguard.counterfactual import (
    InterventionPlan,
    ReplaySwitch,
    ReplayThenLiveManagerClient,
    ReplayThenLiveWorkerBackend,
    extract_replay_manifest,
    load_replay_manifest,
    verify_parent_artifacts,
    write_replay_manifest,
)
from stateguard.counterfactual.replay import replay_report
from stateguard.counterfactual.hashing import source_tree_digest
from stateguard.harness.blind_view import assert_blind
from stateguard.adapters.longds.messages import extract_answer, extract_python, reasoning_text
from stateguard.providers.openai_compatible import OpenAICompatibleClient
from stateguard.validation.models import ManagerAction, ManagerDecision

from .adapter import CorpusAdapter
from .artifacts import write_json
from .runner import preflight_dsgym


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a completed Worker prefix in a fresh official runtime, replace "
            "one Worker action, and continue Worker and Manager live"
        )
    )
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--intervention", type=Path, default=None)
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="replay the complete recorded tape without any paid model calls",
    )
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--dsgym-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", default="counterfactual")
    parser.add_argument("--manager-url", default="http://localhost:5000")

    parser.add_argument("--worker-model", default=None)
    parser.add_argument("--worker-backend", default="litellm")
    parser.add_argument("--worker-api-base", default=None)
    parser.add_argument("--worker-api-key", default=None)
    parser.add_argument("--worker-max-steps", type=int, default=None)
    parser.add_argument("--worker-max-tokens", type=int, default=None)
    parser.add_argument("--worker-timeout", type=float, default=None)
    parser.add_argument("--worker-max-retries", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--review-cadence", type=int, default=None)

    parser.add_argument("--manager-model", default=None)
    parser.add_argument("--manager-api-base", default=None)
    parser.add_argument("--manager-api-key", default=None)
    parser.add_argument("--manager-max-steps", type=int, default=8)
    parser.add_argument("--manager-timeout", type=float, default=300.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.replay_only == (args.intervention is not None):
        raise ValueError("choose exactly one of --replay-only or --intervention")

    parent_run = args.parent_run.expanduser().resolve(strict=True)
    manifest = (
        load_replay_manifest(args.manifest)
        if args.manifest is not None
        else extract_replay_manifest(parent_run)
    )
    if Path(manifest.parent_run).resolve() != parent_run:
        raise ValueError("manifest parent_run does not match --parent-run")
    verify_parent_artifacts(manifest)
    if (
        manifest.metadata.get("worker_replayable_required") is not True
        and args.replay_only
    ):
        raise ValueError(
            "complete clean replay requires a Worker-replayable manifest"
        )
    current_fingerprint = source_tree_digest(Path(__file__).resolve().parents[2])
    if manifest.metadata.get("runtime_fingerprint") != current_fingerprint:
        raise ValueError(
            "StateGuard runtime changed after replay manifest creation; "
            "regenerate the parent replay manifest"
        )
    intervention = (
        None
        if args.replay_only
        else InterventionPlan.from_dict(_read_object(args.intervention))
    )
    if intervention is not None:
        assert_blind(intervention.to_dict())
        _validate_worker_action(intervention.replacement_action, "replacement_action")
        _validate_manager_sequence(intervention.forced_manager_responses)
        for index, response in enumerate(intervention.forced_repair_responses, 1):
            _validate_worker_action(response, f"forced_repair_responses[{index}]")
        _validate_post_repair_manager_sequence(
            intervention.forced_post_repair_manager_responses
        )

    parent_worker_model = str(manifest.metadata.get("worker_model") or "")
    worker_model = args.worker_model or parent_worker_model or "replayed-worker"
    parent_worker_steps = int(manifest.metadata.get("worker_budget_per_unit") or 30)
    worker_steps = args.worker_max_steps or parent_worker_steps
    parent_review_cadence = int(manifest.metadata.get("review_cadence") or 3)
    review_cadence = args.review_cadence or parent_review_cadence
    if args.worker_max_steps is not None and worker_steps != parent_worker_steps:
        raise ValueError("Worker budget must match the parent run")
    if args.review_cadence is not None and review_cadence != parent_review_cadence:
        raise ValueError("review cadence must match the parent run")
    if (
        not args.replay_only
        and parent_worker_model
        and worker_model != parent_worker_model
    ):
        raise ValueError("live Worker model must match the parent run")
    if not args.replay_only:
        _require_live_configuration(args)

    adapter = CorpusAdapter(
        source=manifest.source,
        corpus_root=args.corpus_root,
        dsgym_root=args.dsgym_root,
        output_root=args.output_dir,
        experiment_name=args.experiment,
        mode=manifest.mode,
        model=worker_model,
        backend_type=args.worker_backend,
        manager_url=args.manager_url,
        max_worker_steps=worker_steps,
        review_cadence=review_cadence,
        temperature=args.temperature,
        api_key=args.worker_api_key or os.environ.get("WORKER_API_KEY"),
        base_url=args.worker_api_base or os.environ.get("WORKER_API_BASE"),
        max_model_len=args.max_model_len,
        worker_max_tokens=args.worker_max_tokens,
        worker_timeout=args.worker_timeout,
        worker_max_retries=args.worker_max_retries,
    )
    adapter._ensure_runtime_components()
    assert adapter.backend_factory is not None
    clean_backend_factory = adapter.backend_factory
    switch = ReplaySwitch()
    worker_wrappers: list[ReplayThenLiveWorkerBackend] = []

    def replay_backend_factory() -> ReplayThenLiveWorkerBackend:
        wrapper = ReplayThenLiveWorkerBackend(
            manifest,
            switch,
            live_backend=None if args.replay_only else clean_backend_factory(),
            intervention=intervention,
        )
        worker_wrappers.append(wrapper)
        return wrapper

    adapter.backend_factory = replay_backend_factory
    live_manager = (
        None
        if args.replay_only
        else OpenAICompatibleClient(
            args.manager_model,
            args.manager_api_base or os.environ["MANAGER_API_BASE"],
            args.manager_api_key or os.environ["MANAGER_API_KEY"],
            timeout=args.manager_timeout,
        )
    )
    manager_wrapper = ReplayThenLiveManagerClient(
        manifest,
        switch,
        live_client=live_manager,
        intervention=intervention,
    )
    manager = StateManagerAgent(
        manager_wrapper,
        max_steps_per_action=args.manager_max_steps,
        max_context_chars=manifest.metadata.get("manager_max_context_chars"),
        reserved_output_chars=int(
            manifest.metadata.get("manager_reserved_output_chars") or 0
        ),
    )
    manager.intervention_observer = manager_wrapper

    tasks = adapter.load_tasks(task_ids=(manifest.task_id,))
    if len(tasks) != 1:
        raise RuntimeError(f"expected one replay task, found {len(tasks)}")
    preflight_dsgym(args.manager_url, required_slots=2)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_name = "clean_replay" if intervention is None else intervention.intervention_id
    run_dir = (
        args.output_dir.expanduser().resolve()
        / "counterfactual"
        / manifest.source
        / manifest.mode
        / manifest.task_id
        / _safe_segment(branch_name)
        / stamp
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    # Do not write counterfactual metadata before the Worker finishes. Some
    # recorded Worker actions inspect the run root with os.walk(parent), and
    # pre-writing replay_manifest.json/intervention.json would make the replay
    # environment observably different from the parent clean run. The runner
    # already holds both objects in memory, so persist them after execution.
    result = adapter.run_task(tasks[0], manager=manager, run_dir=run_dir)
    write_replay_manifest(run_dir / "replay_manifest.json", manifest)
    if intervention is not None:
        write_json(run_dir / "intervention.json", intervention.to_dict())
    if len(worker_wrappers) != 1:
        raise RuntimeError(f"expected one Worker replay wrapper, found {len(worker_wrappers)}")
    report = replay_report(worker_wrappers[0], manager_wrapper)
    invalid_reasons = _branch_invalid_reasons(
        manifest, intervention, result, report
    )
    branch_manager_failures = sum(
        len(state_result.manager_failures)
        for state_result in result.stateguard_results
    )
    report.update(
        {
            "parent_run": manifest.parent_run,
            "branch_run": str(run_dir),
            "mode": "clean_replay" if intervention is None else "counterfactual",
            "task_success": bool(result.trajectory.get("success")),
            "task_error": result.error,
            "branch_valid": not invalid_reasons,
            "parent_manager_quality": manifest.metadata.get("manager_quality"),
            "branch_manager_failures": branch_manager_failures,
            "invalid_reasons": invalid_reasons,
        }
    )
    lineage = {
        "parent_run": manifest.parent_run,
        "branch_run": str(run_dir),
        "variant": _lineage_variant(intervention),
        "intervention": None if intervention is None else intervention.to_dict(),
        "branch_valid": not invalid_reasons,
    }
    write_json(run_dir / "counterfactual_lineage.json", lineage)
    write_json(run_dir / "replay_report.json", report)
    print(f"RESULT >>> {run_dir}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if not invalid_reasons else 1


def _replay_mismatched(result) -> bool:
    """Report whether the clean prefix stopped matching the recorded parent.

    The harness is deliberately fail-open for the Manager, so a
    ReplayMismatchError is recorded like any other Manager failure and the
    Worker keeps going. That is right for a protocol or tool failure, whose
    quality is independent of whether the Worker can be intervened on, but a
    mismatch means the prefix never reproduced the parent at all. It can only
    be raised before the switch activates, because both replay wrappers go
    live the moment it flips, so its presence always invalidates the branch.
    """
    return any(
        failure.error_type == "ReplayMismatchError"
        for state_result in result.stateguard_results
        for failure in state_result.manager_failures
    )


def _branch_invalid_reasons(manifest, intervention, result, report) -> list[str]:
    reasons: list[str] = []
    if result.error is not None:
        reasons.append(f"task_error:{result.error}")
    if _replay_mismatched(result):
        reasons.append("replay_prefix_mismatch_before_intervention")
    if intervention is None:
        if not result.trajectory.get("success"):
            reasons.append("clean_replay_task_was_not_successful")
        if report["switch_activated"]:
            reasons.append("clean_replay_activated_switch")
        if report["worker"]["recorded_calls_consumed"] != len(manifest.worker_calls):
            reasons.append("clean_replay_did_not_consume_all_worker_calls")
        if report["manager"]["recorded_calls_consumed"] != len(manifest.manager_calls):
            reasons.append("clean_replay_did_not_consume_all_manager_calls")
    else:
        if not report["switch_activated"]:
            reasons.append("intervention_target_was_not_reached")
        if report["manager"]["forced_manager_responses_remaining"]:
            reasons.append("forced_manager_responses_were_not_consumed")
        if report["manager"]["forced_manager_sequence_error"]:
            reasons.append("forced_manager_response_was_rejected")
        if (
            intervention.forced_manager_responses
            and report["manager"]["forced_manager_repairs_applied"] != 1
        ):
            reasons.append("forced_manager_repair_was_not_applied")
        if report["worker"]["forced_repairs_remaining"]:
            reasons.append("forced_repair_responses_were_not_consumed")
        if report["manager"]["forced_post_repair_manager_responses_remaining"]:
            reasons.append("forced_post_repair_manager_responses_were_not_consumed")
        if report["worker"]["pending_repair_requests"]:
            reasons.append("applied_repair_did_not_reach_a_worker_retry")
    return reasons


def _validate_worker_action(response: str, label: str) -> None:
    has_code = extract_python(response) is not None
    has_answer = extract_answer(response) is not None
    if has_code == has_answer:
        raise ValueError(
            f"{label} must contain exactly one official <python> or <answer> action"
        )
    if not reasoning_text(response).strip():
        raise ValueError(f"{label} must contain a non-empty <reasoning> block")


def _lineage_variant(intervention: InterventionPlan | None) -> str:
    if intervention is None:
        return "clean_replay"
    has_manager = bool(intervention.forced_manager_responses)
    has_worker = bool(intervention.forced_repair_responses)
    if has_manager and has_worker:
        return "full_repair_demonstration"
    if has_manager:
        return "manager_repair_demonstration"
    if has_worker:
        return "worker_repair_demonstration"
    return "autonomous_counterfactual"


def _validate_manager_sequence(responses: tuple[str, ...]) -> None:
    if not responses:
        return

    last_decision: ManagerDecision | None = None
    for index, response in enumerate(responses, 1):
        label = f"forced_manager_responses[{index}]"
        try:
            action = parse_action(response)
        except Exception as exc:
            raise ValueError(
                f"{label} must be a valid Manager Tool or control action: {exc}"
            ) from exc
        if not action.reasoning.strip():
            raise ValueError(f"{label} requires non-empty reasoning")

        decision = None
        if action.kind == "tool":
            if not (action.tool_name or "").strip():
                raise ValueError(f"{label} requires a non-empty tool name")
        elif action.kind == "control":
            try:
                decision = ManagerDecision.from_dict(
                    parse_json_object(action.answer or "")
                )
            except Exception as exc:
                raise ValueError(
                    f"{label} contains an invalid Manager control decision: {exc}"
                ) from exc
            if decision.action is ManagerAction.REPAIR and index != len(responses):
                raise ValueError("forced Manager REPAIR must end the demonstration")
        else:
            raise ValueError(f"{label} must use type=tool or type=control")
        if index == len(responses):
            last_decision = decision

    if last_decision is None or last_decision.action is not ManagerAction.REPAIR:
        raise ValueError("forced Manager responses must end with a REPAIR control action")


def _validate_post_repair_manager_sequence(responses: tuple[str, ...]) -> None:
    if not responses:
        return

    last_decision: ManagerDecision | None = None
    for index, response in enumerate(responses, 1):
        label = f"forced_post_repair_manager_responses[{index}]"
        try:
            action = parse_action(response)
        except Exception as exc:
            raise ValueError(
                f"{label} must be a valid Manager Tool or control action: {exc}"
            ) from exc
        if not action.reasoning.strip():
            raise ValueError(f"{label} requires non-empty reasoning")
        if action.kind == "tool":
            if not (action.tool_name or "").strip():
                raise ValueError(f"{label} requires a non-empty tool name")
            last_decision = None
            continue
        if action.kind != "control":
            raise ValueError(f"{label} must use type=tool or type=control")
        try:
            decision = ManagerDecision.from_dict(
                parse_json_object(action.answer or "")
            )
        except Exception as exc:
            raise ValueError(
                f"{label} contains an invalid Manager control decision: {exc}"
            ) from exc
        if decision.action is ManagerAction.REPAIR:
            raise ValueError("post-repair Manager sequence cannot issue another REPAIR")
        last_decision = decision

    if last_decision is None or last_decision.action is not ManagerAction.COMMIT_STATE:
        raise ValueError(
            "forced post-repair Manager responses must end with COMMIT_STATE"
        )


def _require_live_configuration(args: argparse.Namespace) -> None:
    if not args.worker_model:
        raise ValueError("counterfactual continuation requires --worker-model")
    if not (args.worker_api_base or os.environ.get("WORKER_API_BASE")):
        raise ValueError("counterfactual continuation requires a Worker API base")
    if not (args.worker_api_key or os.environ.get("WORKER_API_KEY")):
        raise ValueError("counterfactual continuation requires a Worker API key")
    if not args.manager_model:
        raise ValueError("counterfactual continuation requires --manager-model")
    if not (args.manager_api_base or os.environ.get("MANAGER_API_BASE")):
        raise ValueError("counterfactual continuation requires a Manager API base")
    if not (args.manager_api_key or os.environ.get("MANAGER_API_KEY")):
        raise ValueError("counterfactual continuation requires a Manager API key")


def _read_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("missing intervention path")
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("intervention file must contain a JSON object")
    return value


def _safe_segment(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    ) or "branch"


if __name__ == "__main__":
    raise SystemExit(main())
