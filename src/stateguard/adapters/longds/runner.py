from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from stateguard.agents.manager import StateManagerAgent
from stateguard.providers.file_handshake import FileHandshakeModelClient
from stateguard.providers.openai_compatible import OpenAICompatibleClient

from .adapter import LongDSAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run official-compatible LongDS with optional StateGuard Manager"
    )
    parser.add_argument("--dsgym-root", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./results"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="litellm")
    parser.add_argument("--manager-url", default="http://localhost:5000")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--worker-max-tokens", type=int, default=None)
    parser.add_argument("--worker-timeout", type=float, default=None)
    parser.add_argument("--worker-max-retries", type=int, default=None)
    parser.add_argument("--task-concurrency", type=int, default=1)
    parser.add_argument("--reset_env_times", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--turn-limit", type=int, default=None)
    parser.add_argument("--manager-model", default=None)
    parser.add_argument("--manager-api-base", default=None)
    parser.add_argument("--manager-api-key", default=None)
    parser.add_argument("--manager-file-dir", type=Path, default=None)
    parser.add_argument("--manager-file-timeout", type=float, default=7200.0)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model", default="deepseek-v4-pro")
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-base-url", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    task_concurrency = getattr(args, "task_concurrency", 1)
    if task_concurrency < 1:
        raise ValueError("--task-concurrency must be at least 1")

    adapter = LongDSAdapter(
        dsgym_root=args.dsgym_root,
        dataset_root=args.dataset_path,
        output_root=args.output_dir,
        model=args.model,
        backend_type=args.backend,
        manager_url=args.manager_url,
        max_steps_per_turn=args.max_steps,
        temperature=args.temperature,
        api_key=args.api_key,
        base_url=args.base_url,
        max_model_len=args.max_model_len,
        worker_max_tokens=getattr(args, "worker_max_tokens", None),
        worker_timeout=args.worker_timeout,
        worker_max_retries=args.worker_max_retries,
        reset_env_times=args.reset_env_times,
    )
    tasks = adapter.load_tasks(
        start_index=args.start_index,
        task_limit=args.task_limit,
        turn_limit=args.turn_limit,
    )

    def run_one(index, task):
        print(f"PROGRESS >>> TASK {index}/{len(tasks)} : {task.task_key}", flush=True)
        try:
            manager = _create_manager(
                args, task.task_key if task_concurrency > 1 else None
            )
            return task, adapter.run_task(task, manager=manager)
        except Exception as exc:
            print(
                f"ERROR >>> TASK {task.task_key} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return task, None

    if task_concurrency == 1:
        completed = [run_one(index, task) for index, task in enumerate(tasks, 1)]
    else:
        completed = []
        with ThreadPoolExecutor(max_workers=task_concurrency) as executor:
            futures = {
                executor.submit(run_one, index, task): task
                for index, task in enumerate(tasks, 1)
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    completed.append(future.result())
                except Exception as exc:
                    print(
                        f"ERROR >>> TASK {task.task_key} failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

    # Judging remains outside task worker threads. It runs only when explicitly
    # requested and does not change within-task StateGuard sequencing.
    for task, result in completed:
        if result is None:
            continue
        print(f"RESULT >>> {result.run_dir}", flush=True)
        if result.trajectory.get("error"):
            print(
                f"ERROR >>> TASK {task.task_key} saved partial results: "
                f"{result.trajectory['error']}",
                flush=True,
            )
        if args.judge:
            adapter.judge(
                result,
                api_key=args.judge_api_key or os.environ.get("JUDGE_API_KEY"),
                base_url=args.judge_base_url or os.environ.get("JUDGE_BASE_URL"),
                judge_model=args.judge_model,
            )
    return 0


def _create_manager(args: argparse.Namespace, task_key: str | None = None) -> StateManagerAgent | None:
    if args.manager_file_dir is not None:
        if args.manager_model or args.manager_api_base or args.manager_api_key:
            raise ValueError(
                "--manager-file-dir is mutually exclusive with Manager API options"
            )
        exchange_dir = args.manager_file_dir
        if task_key is not None:
            safe_task_key = "".join(
                character if character.isalnum() or character in "-_." else "_"
                for character in task_key
            )
            exchange_dir = exchange_dir / safe_task_key
        return StateManagerAgent(
            FileHandshakeModelClient(
                exchange_dir,
                timeout_seconds=args.manager_file_timeout,
            )
        )
    if not args.manager_model:
        return None
    if not args.manager_api_base or not args.manager_api_key:
        raise ValueError(
            "--manager-model requires --manager-api-base and --manager-api-key"
        )
    return StateManagerAgent(
        OpenAICompatibleClient(
            args.manager_model,
            args.manager_api_base,
            args.manager_api_key,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
