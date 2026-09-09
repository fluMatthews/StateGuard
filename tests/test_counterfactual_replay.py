from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateguard.core.models import Message
from stateguard.counterfactual.hashing import messages_digest
from stateguard.counterfactual.manifest import (
    extract_replay_manifest,
    load_replay_manifest,
    verify_parent_artifacts,
    write_replay_manifest,
)
from stateguard.counterfactual.models import (
    InterventionPlan,
    ManagerReplayCall,
    ReplayManifest,
    WorkerReplayCall,
)
from stateguard.counterfactual.replay import (
    ForcedManagerSequenceError,
    ReplayMismatchError,
    ReplaySwitch,
    ReplayThenLiveManagerClient,
    ReplayThenLiveWorkerBackend,
    replay_report,
)
from stateguard.providers.base import ModelResponse


class FakeWorkerBackend:
    def __init__(self, response: str = "live-worker") -> None:
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, conversation: list[dict[str, str]]) -> str:
        self.calls.append(list(conversation))
        return self.response


class FakeManagerClient:
    def __init__(self, response: str = "live-manager") -> None:
        self.response = response
        self.calls: list[tuple[list[Message], list[dict]]] = []

    def complete(self, messages: list[Message], tools: list[dict]) -> ModelResponse:
        self.calls.append((list(messages), list(tools)))
        return ModelResponse(self.response, "live-reasoning")


def test_worker_injection_switches_both_agents_live_and_drops_clean_suffix() -> None:
    first_input = [{"role": "user", "content": "question"}]
    second_input = first_input + [
        {"role": "assistant", "content": "clean-1"},
        {"role": "user", "content": "observation-1"},
    ]
    third_clean_input = second_input + [
        {"role": "assistant", "content": "clean-2"},
        {"role": "user", "content": "clean-observation-2"},
    ]
    manager_input = [Message("system", "manager"), Message("user", "review")]
    manifest = _manifest(
        worker_calls=(
            WorkerReplayCall(1, "task/turn_1", 1, messages_digest(first_input), "clean-1"),
            WorkerReplayCall(2, "task/turn_1", 2, messages_digest(second_input), "clean-2"),
            WorkerReplayCall(3, "task/turn_1", 3, messages_digest(third_clean_input), "clean-3"),
        ),
        manager_calls=(
            ManagerReplayCall(1, "task/turn_1", "TURN_END", messages_digest(manager_input), "recorded-manager"),
        ),
    )
    plan = InterventionPlan(
        "fault-1", "task/turn_1", 2, "counterfactual-2"
    )
    switch = ReplaySwitch()
    live_worker = FakeWorkerBackend()
    live_manager = FakeManagerClient()
    worker = ReplayThenLiveWorkerBackend(
        manifest, switch, live_backend=live_worker, intervention=plan
    )
    manager = ReplayThenLiveManagerClient(
        manifest, switch, live_client=live_manager
    )

    assert manager.complete(manager_input, []).content == "recorded-manager"
    assert worker.generate(first_input) == "clean-1"
    assert worker.generate(second_input) == "counterfactual-2"
    assert switch.activated

    divergent_input = second_input + [
        {"role": "assistant", "content": "counterfactual-2"},
        {"role": "user", "content": "different-observation"},
    ]
    assert worker.generate(divergent_input) == "live-worker"
    assert manager.complete([Message("user", "counterfactual review")], []).content == "live-manager"
    assert len(live_worker.calls) == 1
    assert len(live_manager.calls) == 1
    assert worker.call_index == 2  # clean call 3 was never consumed
    report = replay_report(worker, manager)
    assert report["worker"]["injected_call_index"] == 2
    assert report["worker"]["live_calls"] == 1
    assert report["manager"]["live_calls"] == 1


def test_forced_repair_response_waits_for_manager_repair() -> None:
    initial = [{"role": "user", "content": "q"}]
    manifest = _manifest(
        worker_calls=(
            WorkerReplayCall(1, "task", 1, messages_digest(initial), "clean"),
        )
    )
    live = FakeWorkerBackend()
    switch = ReplaySwitch()
    worker = ReplayThenLiveWorkerBackend(
        manifest,
        switch,
        live_backend=live,
        intervention=InterventionPlan(
            "repair-prefix",
            "task",
            1,
            "fault",
            forced_repair_responses=("verified-repair",),
        ),
    )
    manager = ReplayThenLiveManagerClient(
        manifest,
        switch,
        live_client=FakeManagerClient(
            '{"type":"control","reasoning":"clear error","answer":{"action":"REPAIR"}}'
        ),
    )
    assert worker.generate(initial) == "fault"
    manager.complete([Message("user", "review injected fault")], [])
    assert switch.pending_repair_requests == 0
    switch.on_repair_applied()
    assert worker.generate([]) == "verified-repair"
    assert worker.generate([]) == "live-worker"
    assert len(live.calls) == 1


def test_forced_manager_sequence_runs_before_live_manager_and_ends_in_repair() -> None:
    initial = [{"role": "user", "content": "q"}]
    manifest = _manifest(
        worker_calls=(WorkerReplayCall(1, "task", 1, messages_digest(initial), "clean"),)
    )
    forced_tool = json.dumps(
        {
            "type": "tool",
            "reasoning": "Check the executed value before repairing.",
            "tool": "check_execution",
            "arguments": {"step_id": 1},
        }
    )
    forced_repair = json.dumps(
        {
            "type": "control",
            "reasoning": "The executed value contradicts the explicit constraint.",
            "answer": {
                "action": "REPAIR",
                "evidence": {
                    "violated_constraints": ["x must be positive"],
                    "evidence": ["step 1 returned x=-1"],
                },
                "error_hint": {
                    "error_variable": ["x"],
                    "faulty_reasoning": (
                        "The recorded value x=-1 violates the stated "
                        "positive-value constraint."
                    ),
                },
            },
        }
    )
    plan = InterventionPlan(
        "manager-demo",
        "task",
        1,
        "fault",
        forced_manager_responses=(forced_tool, forced_repair),
    )
    from stateguard.adapters.corpus.counterfactual_runner import (
        _validate_manager_sequence,
    )

    _validate_manager_sequence(plan.forced_manager_responses)
    switch = ReplaySwitch()
    live_worker = FakeWorkerBackend()
    live_manager = FakeManagerClient()
    worker = ReplayThenLiveWorkerBackend(
        manifest, switch, live_backend=live_worker, intervention=plan
    )
    manager = ReplayThenLiveManagerClient(
        manifest, switch, live_client=live_manager, intervention=plan
    )

    assert worker.generate(initial) == "fault"
    first = manager.complete([Message("user", "review")], [])
    assert first.content == forced_tool
    assert first.metadata == {
        "forced_teacher": True,
        "intervention_id": "manager-demo",
    }
    tool_result = Message(
        "user",
        '<tool_result>{"tool":"check_execution","ok":true,"output":"verified"}</tool_result>',
    )
    assert manager.complete([tool_result], []).content == forced_repair
    manager.on_manager_control_accepted("REPAIR")
    manager.on_repair_applied()
    assert worker.generate([]) == "live-worker"
    assert manager.complete([Message("user", "post-repair")], []).content == "live-manager"

    report = replay_report(worker, manager)
    assert report["manager"]["forced_manager_calls"] == 2
    assert report["manager"]["forced_manager_responses_accepted"] == 2
    assert report["manager"]["forced_manager_repair_calls"] == 1
    assert report["manager"]["forced_manager_repairs_applied"] == 1
    assert report["manager"]["forced_manager_responses_remaining"] == 0
    assert report["manager"]["live_calls"] == 1


def test_post_repair_manager_sequence_stabilizes_through_commit_before_live() -> None:
    initial = [{"role": "user", "content": "q"}]
    manifest = _manifest(
        worker_calls=(WorkerReplayCall(1, "task", 1, messages_digest(initial), "clean"),)
    )
    repair = json.dumps({"type":"control","reasoning":"repair conflict","answer":{"action":"REPAIR","analytical_evidence":{"violated_constraints":["x positive"],"evidence":["x=-1"]},"error_hint":{"error_variable":["x"],"faulty_reasoning":"x=-1 violates the constraint"}}})
    update = json.dumps({"type":"control","reasoning":"write corrected result","answer":{"action":"UPDATE_STATE","state_update":{"used_variables":[{"name":"x","value":1}],"conclusions":["x is positive"]}}})
    finalize = json.dumps({"type":"control","reasoning":"relation remains valid","answer":{"action":"FINALIZE_RELATIONS","relation_finalization":{"mode":"confirm"}}})
    commit = json.dumps({"type":"control","reasoning":"corrected state is verified","answer":{"action":"COMMIT_STATE"}})
    plan = InterventionPlan(
        "stable-demo", "task", 1, "fault",
        forced_manager_responses=(repair,),
        forced_repair_responses=("verified-repair",),
        forced_post_repair_manager_responses=(update, finalize, commit),
    )
    from stateguard.adapters.corpus.counterfactual_runner import (
        _validate_manager_sequence, _validate_post_repair_manager_sequence,
    )
    _validate_manager_sequence(plan.forced_manager_responses)
    _validate_post_repair_manager_sequence(plan.forced_post_repair_manager_responses)
    switch = ReplaySwitch()
    worker = ReplayThenLiveWorkerBackend(manifest, switch, live_backend=FakeWorkerBackend(), intervention=plan)
    manager = ReplayThenLiveManagerClient(manifest, switch, live_client=FakeManagerClient(), intervention=plan)

    assert worker.generate(initial) == "fault"
    assert manager.complete([Message("user", "review")], []).content == repair
    manager.on_manager_control_accepted("REPAIR")
    manager.on_repair_applied()
    assert worker.generate([]) == "verified-repair"
    assert manager.complete([Message("user", "post-repair")], []).content == update
    manager.on_manager_control_accepted("UPDATE_STATE")
    assert manager.complete([Message("user", "updated")], []).content == finalize
    manager.on_manager_control_accepted("FINALIZE_RELATIONS")
    assert manager.complete([Message("user", "finalized")], []).content == commit
    manager.on_manager_control_accepted("COMMIT_STATE")
    assert manager.complete([Message("user", "next turn")], []).content == "live-manager"

    report = replay_report(worker, manager)
    assert report["manager"]["forced_manager_responses_accepted"] == 4
    assert report["manager"]["forced_post_repair_manager_calls"] == 3
    assert report["manager"]["forced_manager_sequence_completed"] is True
    assert report["manager"]["forced_manager_responses_remaining"] == 0
    assert report["manager"]["live_calls"] == 1


def test_rejected_forced_control_does_not_shift_the_sequence() -> None:
    initial = [{"role": "user", "content": "q"}]
    manifest = _manifest(
        worker_calls=(WorkerReplayCall(1, "task", 1, messages_digest(initial), "clean"),)
    )
    abstain = json.dumps(
        {"type": "control", "reasoning": "stop this draft", "answer": {"action": "ABSTAIN"}}
    )
    repair = json.dumps(
        {
            "type": "control",
            "reasoning": "repair a concrete conflict",
            "answer": {
                "action": "REPAIR",
                "evidence": {
                    "violated_constraints": ["x must be positive"],
                    "evidence": ["x=-1"],
                },
                "error_hint": {
                    "error_variable": ["x"],
                    "faulty_reasoning": "x=-1 violates the explicit positive constraint.",
                },
            },
        }
    )
    plan = InterventionPlan(
        "reject-demo",
        "task",
        1,
        "fault",
        forced_manager_responses=(abstain, repair),
    )
    switch = ReplaySwitch()
    worker = ReplayThenLiveWorkerBackend(
        manifest, switch, live_backend=FakeWorkerBackend(), intervention=plan
    )
    manager = ReplayThenLiveManagerClient(
        manifest, switch, live_client=FakeManagerClient(), intervention=plan
    )

    assert worker.generate(initial) == "fault"
    assert manager.complete([Message("user", "review")], []).content == abstain
    manager.on_manager_control_rejected("ABSTAIN", "preflight: lifecycle rejected it")
    with pytest.raises(ForcedManagerSequenceError, match="lifecycle rejected"):
        manager.complete([Message("user", "retry")], [])

    report = replay_report(worker, manager)
    assert report["manager"]["forced_manager_calls"] == 1
    assert report["manager"]["forced_manager_responses_accepted"] == 0
    assert report["manager"]["forced_manager_responses_remaining"] == 2
    assert report["manager"]["forced_manager_rejections"] == 1
    assert report["manager"]["forced_manager_sequence_error"] is not None


def test_replay_rejects_input_drift_before_intervention() -> None:
    manifest = _manifest(
        worker_calls=(
            WorkerReplayCall(
                1,
                "task",
                1,
                messages_digest([{"role": "user", "content": "expected"}]),
                "clean",
            ),
        )
    )
    worker = ReplayThenLiveWorkerBackend(manifest, ReplaySwitch())
    with pytest.raises(ReplayMismatchError, match="Worker input mismatch"):
        worker.generate([{"role": "user", "content": "drifted"}])


def test_runtime_worker_data_paths_are_normalized() -> None:
    parent = [
        {
            "role": "user",
            "content": "read /tmp/parent/run/worker_data/table.csv",
        }
    ]
    branch = [
        {
            "role": "user",
            "content": "read /tmp/branch/run/worker_data/table.csv",
        }
    ]
    assert messages_digest(parent) == messages_digest(branch)


def test_manifest_extraction_round_trip_and_parent_verification(tmp_path: Path) -> None:
    run_dir = tmp_path / "clean"
    run_dir.mkdir()
    unit_id = "dsbench_v1/example"
    conversation = [
        {"role": "system", "content": "worker"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "action-1"},
        {"role": "user", "content": "observation"},
        {"role": "assistant", "content": "answer"},
    ]
    _write(
        run_dir / "trajectory.json",
        {
            "source": "dsbench_v1",
            "mode": "single_query",
            "task_id": "example",
            "model": "worker-model",
            "success": True,
            "error": None,
            "conversation": conversation,
            "units": [
                {
                    "unit_id": unit_id,
                    "completed": True,
                    "degraded": False,
                    "solution": "answer",
                    "worker_budget_exhausted": False,
                    "sft_eligible": True,
                    "worker_steps": 2,
                    "trajectory": conversation,
                }
            ],
        },
    )
    _write(
        run_dir / "manager_session.json",
        {
            "messages": [
                {"role": "system", "content": "manager", "metadata": {}},
                {"role": "user", "content": "controller", "metadata": {}},
                {
                    "role": "user",
                    "content": "<manager_observation>{}</manager_observation>",
                    "metadata": {
                        "manager_block_start": True,
                        "event_type": "STEP_WINDOW",
                        "task_id": unit_id,
                    },
                },
                {
                    "role": "assistant",
                    "content": '{"type":"control","reasoning":"ok","answer":"{\\"action\\":\\"PASS\\"}"}',
                    "metadata": {"reasoning": "hidden"},
                },
            ]
        },
    )
    _write(run_dir / "manager_failures.json", [])
    _write(
        run_dir / "run_metadata.json",
        {
            "source": "dsbench_v1",
            "mode": "single_query",
            "task_id": "example",
            "worker_budget_per_unit": 30,
            "review_cadence": 3,
        },
    )

    manifest = extract_replay_manifest(run_dir)
    assert len(manifest.worker_calls) == 2
    assert len(manifest.manager_calls) == 1
    path = tmp_path / "manifest.json"
    write_replay_manifest(path, manifest)
    loaded = load_replay_manifest(path)
    assert loaded == manifest
    verify_parent_artifacts(loaded)

    (run_dir / "run_metadata.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_parent_artifacts(loaded)


def test_manager_failures_do_not_block_worker_intervention(tmp_path: Path) -> None:
    # Manager quality is post-hoc metadata, not Worker replay eligibility.
    test_manifest_extraction_round_trip_and_parent_verification(tmp_path)
    run_dir = tmp_path / "clean"
    _write(run_dir / "run_metadata.json", {"source": "dsbench_v1", "mode": "single_query", "task_id": "example"})
    _write(run_dir / "manager_failures.json", [{"error_type": "ProtocolError"}])
    trajectory = json.loads((run_dir / "trajectory.json").read_text(encoding="utf-8"))
    trajectory["units"][0]["degraded"] = True
    trajectory["units"][0]["sft_eligible"] = False
    _write(run_dir / "trajectory.json", trajectory)
    manifest = extract_replay_manifest(run_dir)
    assert manifest.metadata["worker_replayable_required"] is True
    assert manifest.metadata["manager_quality"]["clean"] is False
    assert "runtime_failures:1" in manifest.metadata["manager_quality"]["issues"]


def _manifest(
    *,
    worker_calls: tuple[WorkerReplayCall, ...] = (),
    manager_calls: tuple[ManagerReplayCall, ...] = (),
) -> ReplayManifest:
    return ReplayManifest(
        parent_run="/tmp/parent",
        source="dsbench_v1",
        mode="single_query",
        task_id="task",
        worker_calls=worker_calls,
        manager_calls=manager_calls,
        parent_files={},
    )


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_stdout_chunking_does_not_change_the_replay_digest():
    """Flush timing splits one print block into several banner-tagged chunks.

    The parent run and a replay of the same code capture identical text, so the
    banner must not decide whether the prefix replays. Content differences still
    have to survive the fold.
    """
    from stateguard.counterfactual.hashing import messages_digest

    def digest(content: str) -> str:
        return messages_digest([{"role": "user", "content": content}])

    chunked = "[stdout] shape: (2648, 10)\n\n[stdout] missing: False\n"
    contiguous = "[stdout] shape: (2648, 10)\nmissing: False\n"
    assert digest(chunked) == digest(contiguous)
    assert digest(chunked) == digest("shape: (2648, 10)\nmissing: False\n")
    assert digest("[stdout] v: 1\n") != digest("[stdout] v: 2\n")
    assert digest("[stdout] a\nb\n") != digest("[stdout] b\na\n")
