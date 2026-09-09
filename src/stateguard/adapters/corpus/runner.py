from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from stateguard.agents.manager import StateManagerAgent
from stateguard.providers.file_handshake import FileHandshakeModelClient
from stateguard.providers.openai_compatible import OpenAICompatibleClient

from .adapter import CorpusAdapter, CorpusRunResult
from .artifacts import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run normalized analytical corpora through StateGuard and export SFT activations"
    )
    parser.add_argument(
        "--source",
        choices=(
            "dsbench_v1",
            "idabench_v2",
            "stateguard_v6",
            "stateguard_medium_single",
        ),
        required=True,
    )
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--dsgym-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", default="default")
    parser.add_argument(
        "--mode", choices=("multi-turn", "single-query"), default=None
    )
    parser.add_argument("--task-id", action="append", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--unit-limit", type=int, default=None)
    parser.add_argument("--task-concurrency", type=int, default=1)

    parser.add_argument("--worker-model", required=True)
    parser.add_argument("--worker-backend", default="litellm")
    parser.add_argument("--worker-api-base", default=None)
    parser.add_argument("--worker-api-key", default=None)
    parser.add_argument("--worker-max-steps", type=int, default=30)
    parser.add_argument("--worker-max-tokens", type=int, default=None)
    parser.add_argument("--worker-timeout", type=float, default=None)
    parser.add_argument("--worker-max-retries", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--manager-url", default="http://localhost:5000")
    parser.add_argument("--review-cadence", type=int, default=3)

    parser.add_argument("--manager-model", default=None)
    parser.add_argument("--manager-api-base", default=None)
    parser.add_argument("--manager-api-key", default=None)
    parser.add_argument("--manager-file-dir", type=Path, default=None)
    parser.add_argument("--manager-file-timeout", type=float, default=7200.0)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="return success when some requested tasks fail; failed units remain excluded from SFT",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.task_concurrency < 1:
        raise ValueError("task concurrency must be positive")
    mode = None if args.mode is None else args.mode.replace("-", "_")
    adapter = CorpusAdapter(
        source=args.source,
        corpus_root=args.corpus_root,
        dsgym_root=args.dsgym_root,
        output_root=args.output_dir,
        experiment_name=args.experiment,
        mode=mode,
        model=args.worker_model,
        backend_type=args.worker_backend,
        manager_url=args.manager_url,
        max_worker_steps=args.worker_max_steps,
        review_cadence=args.review_cadence,
        temperature=args.temperature,
        api_key=args.worker_api_key or os.environ.get("WORKER_API_KEY"),
        base_url=args.worker_api_base or os.environ.get("WORKER_API_BASE"),
        max_model_len=args.max_model_len,
        worker_max_tokens=args.worker_max_tokens,
        worker_timeout=args.worker_timeout,
        worker_max_retries=args.worker_max_retries,
    )
    tasks = adapter.load_tasks(
        task_ids=tuple(args.task_id) if args.task_id else None,
        start_index=args.start_index,
        task_limit=args.task_limit,
        unit_limit=args.unit_limit,
    )
    if not tasks:
        print("ERROR >>> no corpus tasks matched the requested selection", flush=True)
        return 2
    manager_enabled = bool(args.manager_model or args.manager_file_dir)
    required_slots = args.task_concurrency * (2 if manager_enabled else 1)
    try:
        preflight_dsgym(args.manager_url, required_slots=required_slots)
    except Exception as exc:
        print(
            f"ERROR >>> DSGym preflight failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return 2

    def run_one(task) -> CorpusRunResult:
        manager = _create_manager(args, task.task_key)
        return adapter.run_task(task, manager=manager)

    completed: list[CorpusRunResult] = []
    raised_failures: list[dict[str, str]] = []
    if args.task_concurrency == 1:
        for index, task in enumerate(tasks, 1):
            print(f"PROGRESS >>> TASK {index}/{len(tasks)} : {task.task_key}", flush=True)
            try:
                completed.append(run_one(task))
            except Exception as exc:
                raised_failures.append(
                    {
                        "task_key": task.task_key,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"ERROR >>> TASK {task.task_key}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
    else:
        with ThreadPoolExecutor(max_workers=args.task_concurrency) as executor:
            futures = {executor.submit(run_one, task): task for task in tasks}
            finished = 0
            for future in as_completed(futures):
                task = futures[future]
                try:
                    completed.append(future.result())
                except Exception as exc:
                    raised_failures.append(
                        {
                            "task_key": task.task_key,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(
                        f"ERROR >>> TASK {task.task_key}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                finished += 1
                print(f"PROGRESS >>> FINISHED {finished}/{len(tasks)}", flush=True)

    export_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregate_records: list[dict] = []
    task_summaries: list[dict] = []
    for result in completed:
        sft_path = result.run_dir / "sft_data_activations.json"
        success = bool(result.trajectory.get("success", False)) and result.error is None
        records: list[dict] = []
        sft_status = "disabled" if not manager_enabled else "skipped_no_activations"
        artifact_error: str | None = None
        result_units = getattr(result, "unit_results", ())
        eligible_units = sum(
            bool(
                row.get(
                    "sft_eligible",
                    row["completed"]
                    and not row["degraded"]
                    and not row.get("worker_budget_exhausted", False),
                )
            )
            for row in result_units
        )
        if manager_enabled:
            if not sft_path.is_file():
                if success:
                    success = False
                    artifact_error = "Manager produced no activation-level SFT artifact"
                sft_status = "skipped_no_activations"
            else:
                try:
                    value = json.loads(sft_path.read_text(encoding="utf-8"))
                    if not isinstance(value, list):
                        raise TypeError("SFT artifact must contain a JSON list")
                    records = value
                    if not records:
                        if success:
                            success = False
                            artifact_error = "Manager produced no accepted SFT activations"
                        sft_status = "no_eligible_unit_activations"
                    else:
                        aggregate_records.extend(records)
                        sft_status = (
                            "accepted" if success else "accepted_partial_units"
                        )
                except Exception as exc:
                    success = False
                    artifact_error = f"{type(exc).__name__}: {exc}"
                    sft_status = "invalid_artifact"
        elif not success:
            sft_status = "skipped_failed_task"
        task_summaries.append(
            {
                "task_key": result.task.task_key,
                "run_dir": str(result.run_dir),
                "success": success,
                "error": result.error,
                "artifact_error": artifact_error,
                "sft_status": sft_status,
                "sft_records": len(records),
                "eligible_units": eligible_units,
                "total_units": len(result_units),
            }
        )
        print(f"RESULT >>> {result.run_dir}", flush=True)

    task_summaries.extend(
        {
            "task_key": failure["task_key"],
            "run_dir": None,
            "success": False,
            "error": failure["error"],
            "artifact_error": None,
            "sft_status": "skipped_failed_task",
            "sft_records": 0,
        }
        for failure in raised_failures
    )
    succeeded_tasks = sum(bool(row["success"]) for row in task_summaries)
    failed_tasks = len(task_summaries) - succeeded_tasks

    export_dir = adapter.run_root / "exports" / export_stamp
    write_json(export_dir / "sft_data_activations.json", aggregate_records)
    write_json(
        export_dir / "run_summary.json",
        {
            "source": args.source,
            "mode": mode,
            "requested_tasks": len(tasks),
            "completed_tasks": len(completed),
            "succeeded_tasks": succeeded_tasks,
            "failed_tasks": failed_tasks,
            "allow_partial": args.allow_partial,
            "sft_records": len(aggregate_records),
            "tasks": task_summaries,
        },
    )
    print(f"SFT_EXPORT >>> {export_dir / 'sft_data_activations.json'}", flush=True)
    if failed_tasks and not args.allow_partial:
        return 1
    return 0


def preflight_dsgym(
    manager_url: str,
    *,
    required_slots: int,
    timeout: float = 15.0,
) -> None:
    """Verify the official local manager and one real Jupyter executor."""
    if required_slots < 1:
        raise ValueError("required DSGym slots must be positive")
    base = manager_url.rstrip("/")
    health = _request_json(f"{base}/health", timeout=timeout)
    if health.get("status") != "ok":
        raise RuntimeError(f"unexpected DSGym health response: {health}")
    status = _request_json(f"{base}/status", timeout=timeout)
    available = int(status.get("available_containers", 0))
    if available < required_slots:
        raise RuntimeError(
            f"DSGym has {available} available slots; {required_slots} are required"
        )

    allocation = _request_json(f"{base}/allocate", method="POST", timeout=timeout)
    container_id = int(allocation["container_id"])
    try:
        ready = _request_json(
            f"{base}/session/{container_id}/ready", timeout=timeout
        )
        if ready.get("ready") is not True:
            raise RuntimeError(f"DSGym executor {container_id} is not ready: {ready}")
        executed = _request_json(
            f"{base}/session/{container_id}/execute",
            method="POST",
            payload={"code": "print('stateguard_preflight_ok')"},
            timeout=timeout,
        )
        if "stateguard_preflight_ok" not in json.dumps(executed):
            raise RuntimeError(
                f"DSGym executor {container_id} failed the execution probe: {executed}"
            )
    finally:
        _request_json(
            f"{base}/deallocate/{container_id}",
            method="POST",
            timeout=max(180.0, timeout),
        )


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    timeout: float,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"DSGym request failed for {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"DSGym returned a non-object response for {url}")
    return value


def _create_manager(args: argparse.Namespace, task_key: str):
    if args.manager_file_dir is not None:
        if args.manager_model or args.manager_api_base or args.manager_api_key:
            raise ValueError("file Manager is mutually exclusive with Manager API options")
        safe_key = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in task_key
        )
        return StateManagerAgent(
            FileHandshakeModelClient(
                args.manager_file_dir / safe_key,
                timeout_seconds=args.manager_file_timeout,
            )
        )
    if not args.manager_model:
        return None
    api_base = args.manager_api_base or os.environ.get("MANAGER_API_BASE")
    api_key = args.manager_api_key or os.environ.get("MANAGER_API_KEY")
    if not api_base or not api_key:
        raise ValueError("Manager API mode requires API base and key")
    return StateManagerAgent(OpenAICompatibleClient(args.manager_model, api_base, api_key))


if __name__ == "__main__":
    raise SystemExit(main())
