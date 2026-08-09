from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from stateguard.agents.manager import StateManagerAgent
from stateguard.providers.file_handshake import FileHandshakeModelClient
from stateguard.providers.openai_compatible import OpenAICompatibleClient

from .adapter import DABstepAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official DABstep Worker with an optional StateGuard Manager"
    )
    parser.add_argument("--dabstep-root", type=Path, required=True)
    parser.add_argument("--official-runner-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("./runs"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--experiment", default="default")
    parser.add_argument("--split", choices=["default", "dev"], default="default")
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--tasks-ids", type=int, nargs="+", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--review-cadence", type=int, default=3)
    parser.add_argument(
        "--runtime-profile",
        choices=["compat-v1", "official"],
        default="compat-v1",
        help="Use the same DABstep Worker runtime profile for baseline and StateGuard.",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--manager-model", default=None)
    parser.add_argument("--manager-api-base", default=None)
    parser.add_argument("--manager-api-key", default=None)
    parser.add_argument("--manager-file-dir", type=Path, default=None)
    parser.add_argument("--manager-file-timeout", type=float, default=7200.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_tasks >= 0 and args.tasks_ids is not None:
        raise ValueError("--max-tasks and --tasks-ids are mutually exclusive")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    adapter = DABstepAdapter(
        dabstep_root=args.dabstep_root,
        official_runner_root=args.official_runner_root,
        output_root=args.output_dir,
        model_id=args.model_id,
        split=args.split,
        experiment_name=args.experiment,
        api_base=args.api_base or os.environ.get("WORKER_API_BASE"),
        api_key=args.api_key or os.environ.get("WORKER_API_KEY"),
        max_worker_steps=args.max_steps,
        review_cadence=args.review_cadence,
        runtime_profile=args.runtime_profile,
    )
    limit = None if args.max_tasks < 0 else args.max_tasks
    tasks = adapter.load_tasks(
        task_ids=args.tasks_ids,
        start_index=args.start_index,
        task_limit=limit,
    )

    def run_one(task):
        manager = _create_manager(args, task.task_id)
        return adapter.run_task(task, manager=manager)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(run_one, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            completed += 1
            try:
                result = future.result()
                print(
                    f"PROGRESS >>> TASK {completed}/{len(tasks)} : {task.task_id} "
                    f"RESULT={result.run_dir}",
                    flush=True,
                )
                if result.error:
                    print(f"ERROR >>> {task.task_id}: {result.error}", flush=True)
            except Exception as exc:
                print(
                    f"ERROR >>> TASK {task.task_id} failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
    return 0


def _create_manager(args: argparse.Namespace, task_id: str) -> StateManagerAgent | None:
    if args.manager_file_dir is not None:
        if args.manager_model or args.manager_api_base or args.manager_api_key:
            raise ValueError("file Manager is mutually exclusive with Manager API options")
        return StateManagerAgent(
            FileHandshakeModelClient(
                args.manager_file_dir / task_id,
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
