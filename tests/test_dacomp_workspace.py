from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stateguard.adapters.dacomp.executor import DACompProbeExecutor
from stateguard.adapters.dacomp.workspace import DACompWorkspace


class DACompWorkspaceTests(unittest.TestCase):
    def test_snapshot_does_not_copy_database_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            database = source / "large_start.duckdb"
            with database.open("wb") as handle:
                handle.truncate(8 * 1024 * 1024)
            (source / "run.py").write_text("print('ok')\n", encoding="utf-8")
            workspace = DACompWorkspace(base / "work", source)
            workspace.prepare()
            (workspace.root / "sql").mkdir()
            (workspace.root / "sql" / "model.sql").write_text("select 1", encoding="utf-8")
            snapshot = workspace.snapshot()
            self.assertIn("large_start.duckdb", snapshot.database_files)
            self.assertNotIn(
                "large_start.duckdb", {relative for relative, _ in snapshot.overrides}
            )
            self.assertLess(sum(len(content) for _, content in snapshot.overrides), 1024)


    def test_probe_reuses_one_read_only_data_copy_and_cleans_it_on_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            (source / "input.sqlite").write_bytes(b"not-a-real-db")
            (source / "note.txt").write_text("public", encoding="utf-8")
            workspace = DACompWorkspace(base / "work", source)
            workspace.prepare()
            executor = DACompProbeExecutor(workspace)
            first = executor.execute(
                "print(Path(data_files['note.txt']).read_text())"
            )
            self.assertTrue(first.ok, first.error)
            self.assertIn("public", first.stdout)
            cache_root = executor._cache_root
            self.assertIsNotNone(cache_root)
            second = executor.execute("print(DATA_ROOT)")
            self.assertTrue(second.ok, second.error)
            self.assertEqual(cache_root, executor._cache_root)
            database = cache_root / "task_data" / "input.sqlite"
            self.assertEqual(database.stat().st_mode & 0o222, 0)
            executor.close()
            self.assertFalse(cache_root.exists())

    def test_restore_reapplies_non_database_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            (source / "note.txt").write_text("original", encoding="utf-8")
            workspace = DACompWorkspace(base / "work", source)
            workspace.prepare()
            (workspace.root / "note.txt").write_text("checkpoint", encoding="utf-8")
            snapshot = workspace.snapshot()
            (workspace.root / "note.txt").write_text("later", encoding="utf-8")
            (workspace.root / "extra.txt").write_text("later", encoding="utf-8")
            workspace.restore(snapshot)
            self.assertEqual(
                (workspace.root / "note.txt").read_text(encoding="utf-8"), "checkpoint"
            )
            self.assertFalse((workspace.root / "extra.txt").exists())


if __name__ == "__main__":
    unittest.main()
