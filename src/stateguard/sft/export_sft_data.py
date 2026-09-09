"""Export one clean Manager SFT file per corpus source.

Run this over a completed corpus run. It reads each task's archived Manager
session, rebuilds the activation records with the shipped exporter, and writes
one JSON array per source.

Two policies differ from a per-task export:

  * The multi-turn unit gate is dropped. It rejects every activation in a turn
    whose Worker never reached an answer, but a Worker that ran out of budget
    does not make the Manager's own state handling wrong -- an ABSTAIN or a
    partial commit there is exactly the behaviour worth learning.
  * Records are filtered by real token count rather than the exporter's
    character budget, because the newest block is always retained and a long
    activation can exceed the window no matter what the budget says.

Blocks that carry no Manager action at all -- a turn's task_unit_initialization
is injected with the same block marker -- are counted separately from genuine
rejections so the report does not read as lost training data.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from stateguard.sft.activation import export_manager_activations

DEFAULT_MAX_TOKENS = 40_000
DEFAULT_TOKENIZER = "/fs/fast/u2024201619/models/Qwen3-8B/tokenizer.json"


def _load_failures(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "manager_failures.json"
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    return list(value.get("failures") or [])


def _record_tokens(record: dict[str, Any], encode) -> int:
    return len(encode("\n".join(m["content"] for m in record["messages"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer)
    encode = lambda text: tokenizer.encode(text).ids  # noqa: E731

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []

    for source_dir in sorted(p for p in args.corpus_root.iterdir() if p.is_dir()):
        records: list[dict[str, Any]] = []
        reasons: Counter[str] = Counter()
        tasks = skipped = rejected = dropped = 0
        for session_path in sorted(source_dir.glob("*/*/*/*/*/manager_session.json")):
            run_dir = session_path.parent
            tasks += 1
            accepted, report = export_manager_activations(
                json.loads(session_path.read_text(encoding="utf-8")),
                _load_failures(run_dir),
            )
            for row in report:
                if row.get("accepted"):
                    continue
                row_reasons = list(row.get("reasons") or [])
                if row_reasons == ["block_does_not_start_with_manager_observation"]:
                    skipped += 1
                    continue
                rejected += 1
                for reason in row_reasons:
                    reasons[reason.split(":")[0].split("[")[0]] += 1
            for record in accepted:
                if _record_tokens(record, encode) > args.max_tokens:
                    dropped += 1
                    continue
                records.append(record)

        out = args.output_dir / f"{source_dir.name}.json"
        out.write_text(
            json.dumps(records, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        summary.append(
            {
                "source": source_dir.name,
                "tasks": tasks,
                "records": len(records),
                "non_activation_blocks_skipped": skipped,
                "rejected_activations": rejected,
                "dropped_over_token_limit": dropped,
                "rejection_reasons": dict(reasons.most_common()),
                "file": str(out),
            }
        )
        print(
            f"{source_dir.name:26s} {tasks:3d} 题  {len(records):5d} 条  "
            f"(跳过伪block {skipped}, 拒绝 {rejected}, 超长丢弃 {dropped})"
        )

    report_path = args.output_dir / "export_summary.json"
    report_path.write_text(
        json.dumps(
            {"max_tokens": args.max_tokens, "sources": summary},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n合计 {sum(s['records'] for s in summary)} 条 -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
