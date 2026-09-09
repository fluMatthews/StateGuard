from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .manifest import extract_replay_manifest, write_replay_manifest


def prepare_manifests(
    runs_root: Path, *, overwrite: bool = False
) -> dict[str, object]:
    """Prepare manifests for runs with complete, replayable Worker trajectories."""
    runs_root = runs_root.expanduser().resolve(strict=True)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    for trajectory_path in sorted(runs_root.rglob("trajectory.json")):
        run_dir = trajectory_path.parent
        if (run_dir / "counterfactual_lineage.json").is_file():
            continue
        output = run_dir / "replay_manifest.json"
        if output.exists() and not overwrite:
            accepted.append(
                {
                    "run_dir": str(run_dir),
                    "manifest": str(output),
                    "status": "existing",
                }
            )
            continue
        try:
            manifest = extract_replay_manifest(run_dir)
            write_replay_manifest(output, manifest)
            accepted.append(
                {
                    "run_dir": str(run_dir),
                    "manifest": str(output),
                    "status": "written",
                    "worker_calls": len(manifest.worker_calls),
                    "manager_calls": len(manifest.manager_calls),
                }
            )
        except Exception as exc:
            rejected.append(
                {
                    "run_dir": str(run_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "runs_root": str(runs_root),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_runs": accepted,
        "rejected_runs": rejected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare replay manifests for completed Worker runs"
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = prepare_manifests(args.runs_root, overwrite=args.overwrite)
    _write_json(args.report, report)
    print(args.report.expanduser().resolve())
    print(f"accepted={report['accepted']} rejected={report['rejected']}")
    return 0


def _write_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
