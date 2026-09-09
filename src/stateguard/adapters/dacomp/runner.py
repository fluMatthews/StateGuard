from __future__ import annotations

import argparse
import os
from pathlib import Path

from stateguard.agents.manager import StateManagerAgent
from stateguard.providers.file_handshake import FileHandshakeModelClient
from stateguard.providers.openai_compatible import OpenAICompatibleClient

from .adapter import DACompAdapter
from .dataset import DACompTrack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DAComp DA-stage1 or DE Impl/Evol with optional StateGuard Manager"
    )
    parser.add_argument("--dacomp-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./results"))
    parser.add_argument(
        "--track",
        choices=[track.value for track in DACompTrack],
        required=True,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="DA model name, or DE official OpenHands llm-config name",
    )
    parser.add_argument("--experiment-name", default="default")
    parser.add_argument("--task-ids", default=None, help="comma-separated official IDs")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--review-cadence", type=int, default=5)
    parser.add_argument("--max-worker-steps", type=int, default=None)
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    # Some endpoints reject any value but their own default (kimi-k2.6:
    # "only 1 is allowed for this model"), so the DA worker temperature is
    # exposed rather than fixed at the official 0.0.
    parser.add_argument("--da-temperature", type=float, default=0.0)
    parser.add_argument("--da-top-p", type=float, default=1.0)
    parser.add_argument("--manager-model", default=None)
    parser.add_argument("--manager-api-base", default=None)
    parser.add_argument("--manager-api-key", default=None)
    parser.add_argument("--manager-file-dir", type=Path, default=None)
    parser.add_argument("--manager-file-timeout", type=float, default=7200.0)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--rubrics-model", default=None)
    parser.add_argument(
        "--de-eval-python",
        default=None,
        help="Python executable containing official DAComp-DE evaluator dependencies",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    adapter = DACompAdapter(
        dacomp_root=args.dacomp_root,
        output_root=args.output_dir,
        track=args.track,
        model=args.model,
        experiment_name=args.experiment_name,
        review_cadence=args.review_cadence,
        max_worker_steps=args.max_worker_steps,
        language=args.language,
        da_temperature=args.da_temperature,
        da_top_p=args.da_top_p,
    )
    task_ids = (
        [item.strip() for item in args.task_ids.split(",") if item.strip()]
        if args.task_ids
        else None
    )
    tasks = adapter.load_tasks(
        task_ids=task_ids,
        start_index=args.start_index,
        task_limit=args.task_limit,
    )
    for index, task in enumerate(tasks, 1):
        print(f"PROGRESS >>> TASK {index}/{len(tasks)} : {task.instance_id}", flush=True)
        try:
            manager = _create_manager(args, task.instance_id)
            result = adapter.run_task(task, manager=manager)
            print(f"RESULT >>> {result.run_dir}", flush=True)
            if result.error:
                print(f"ERROR >>> {task.instance_id}: {result.error}", flush=True)
            if args.judge:
                judge_kwargs = {}
                if task.track is DACompTrack.DA_STAGE1:
                    judge_kwargs["rubrics_model"] = args.rubrics_model
                    judge_kwargs["language"] = args.language
                elif args.de_eval_python:
                    judge_kwargs["python_executable"] = args.de_eval_python
                judged = adapter.judge(result, **judge_kwargs)
                print(f"JUDGE >>> {task.instance_id}: {judged}", flush=True)
        except Exception as exc:
            print(
                f"ERROR >>> TASK {task.instance_id} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
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
    return StateManagerAgent(
        OpenAICompatibleClient(args.manager_model, api_base, api_key)
    )


if __name__ == "__main__":
    raise SystemExit(main())
