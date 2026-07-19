"""Episode + compaction-summary persistence across restarts (Phase B7)."""

import tempfile
from pathlib import Path

from cascade.episodes import generate_episode
from cascade.history.database import HistoryDB


def _db(tmpdir):
    return HistoryDB(str(Path(tmpdir) / "history.db"))


class TestContextPersistence:
    def test_round_trip_episodes_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _db(tmpdir)
            session = db.create_session(provider="claude")
            eps = [
                generate_episode(
                    "Fix cascade/state.py", "Fixed the reducer.", "claude",
                    tokens=1200, source="compaction",
                ),
                generate_episode("Live turn", "Replied.", "gemini"),
            ]
            db.save_context(
                session["id"], eps, "1. Primary Request: parser work",
                compacted_through=12,
            )

            loaded, summary, compacted_through = db.load_context(session["id"])
            assert compacted_through == 12
            assert summary == "1. Primary Request: parser work"
            assert [e.objective for e in loaded] == [e.objective for e in eps]
            assert loaded[0].source == "compaction"
            assert loaded[0].artifacts == eps[0].artifacts
            assert loaded[0].tokens_consumed == 1200
            assert loaded[1].provider == "gemini"
            db.close()

    def test_replace_all_semantics_mirror_pruning(self):
        """Stored episodes always mirror the in-memory list -- pruning included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _db(tmpdir)
            session = db.create_session()
            first = [generate_episode(f"t{i}", "done", "claude") for i in range(5)]
            db.save_context(session["id"], first, "")
            db.save_context(session["id"], first[-2:], "kept summary")

            loaded, summary, _ = db.load_context(session["id"])
            assert len(loaded) == 2
            assert summary == "kept summary"
            db.close()

    def test_load_context_empty_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _db(tmpdir)
            session = db.create_session()
            loaded, summary, compacted_through = db.load_context(session["id"])
            assert loaded == []
            assert summary == ""
            assert compacted_through == 0
            db.close()

    def test_migration_is_idempotent_and_versioned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "history.db")
            db = HistoryDB(path)
            version = db._conn.execute("PRAGMA user_version").fetchone()[0]
            assert version >= 1
            db.close()
            # Re-opening must not fail or re-run migrations destructively
            db2 = HistoryDB(path)
            session = db2.create_session()
            db2.save_context(session["id"], [generate_episode("x", "y", "claude")], "s")
            loaded, _, _ = db2.load_context(session["id"])
            assert len(loaded) == 1
            db2.close()

    def test_deleting_session_cascades_episodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _db(tmpdir)
            session = db.create_session()
            db.save_context(session["id"], [generate_episode("x", "y", "claude")], "")
            db.delete_session(session["id"])
            rows = db._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            assert rows == 0
            db.close()
