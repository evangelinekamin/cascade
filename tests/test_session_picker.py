"""Session picker: directory-scoped recent sessions + the modal that lists them.

Covers three things the /resume-needs-a-menu change introduced:
  1. The cwd schema migration is crash-safe -- a fresh DB and a DB built at the
     OLD (pre-cwd) schema version both end up with episodes, session_context AND
     the cwd column. This is the round-trip that would have caught the round-1
     data loss (a second migration runner that skipped those tables).
  2. list_sessions_for_cwd filters to the directory, keeps unknown-dir rows
     reachable, and reports message counts.
  3. SessionPickerScreen actually renders in a pilot and dismisses the chosen
     id (arrow + enter) or None (escape).
"""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from cascade.history.database import HistoryDB, _SCHEMA_VERSION
from cascade.screens.session_picker import (
    SessionPickerScreen,
    _format_row,
    _relative_time,
)


# ---------------------------------------------------------------------------
# Migration round-trip (the data-loss guard)
# ---------------------------------------------------------------------------

def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def _sessions_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(sessions)").fetchall()
    return {r[1] for r in rows}


def test_fresh_db_has_all_tables_and_cwd_column(tmp_path):
    db = HistoryDB(db_path=str(tmp_path / "fresh.db"))
    try:
        conn = db._conn
        assert {"episodes", "session_context"} <= _table_names(conn)
        assert "cwd" in _sessions_columns(conn)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION
    finally:
        db.close()


def _build_old_v2_schema(path: str) -> str:
    """A DB frozen at schema version 2 (episodes + session_context, no cwd)."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '',
            provider TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE messages (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
            role TEXT NOT NULL, content TEXT NOT NULL, timestamp TEXT NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0, metadata TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE episodes (
            id TEXT NOT NULL, session_id TEXT NOT NULL REFERENCES sessions(id),
            timestamp REAL NOT NULL, provider TEXT NOT NULL DEFAULT '',
            objective TEXT NOT NULL DEFAULT '', actions TEXT NOT NULL DEFAULT '[]',
            outcome TEXT NOT NULL DEFAULT '', artifacts TEXT NOT NULL DEFAULT '[]',
            tokens INTEGER NOT NULL DEFAULT 0, turn_count INTEGER NOT NULL DEFAULT 1,
            source TEXT NOT NULL DEFAULT 'live', PRIMARY KEY (session_id, id)
        );
        CREATE TABLE session_context (
            session_id TEXT PRIMARY KEY REFERENCES sessions(id),
            compaction_summary TEXT NOT NULL DEFAULT '',
            compacted_through INTEGER NOT NULL DEFAULT 0,
            compaction_boundary TEXT NOT NULL DEFAULT '',
            compaction_count INTEGER NOT NULL DEFAULT 0
        );
    """)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO sessions (id, title, provider, model, created_at, updated_at) "
        "VALUES ('legacy-row', 'Old chat', 'claude', 'claude-opus-4-8', ?, ?)",
        (now, now),
    )
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()
    return path


def test_migrating_old_v2_db_adds_cwd_and_keeps_context_tables(tmp_path):
    path = _build_old_v2_schema(str(tmp_path / "legacy.db"))

    db = HistoryDB(db_path=path)  # triggers _migrate()
    try:
        conn = db._conn
        # The data-loss guard: episodes + session_context survive the upgrade,
        # AND the new cwd column exists.
        assert {"episodes", "session_context"} <= _table_names(conn)
        assert "cwd" in _sessions_columns(conn)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == _SCHEMA_VERSION

        # Pre-existing row was backfilled with '' (unknown), not dropped.
        legacy = db.get_session("legacy-row")
        assert legacy is not None
        assert legacy["cwd"] == ""
        assert legacy["title"] == "Old chat"
    finally:
        db.close()


def test_reopening_migrated_db_is_idempotent(tmp_path):
    """Opening an already-migrated DB must not re-run ALTER (duplicate column)."""
    path = _build_old_v2_schema(str(tmp_path / "twice.db"))
    HistoryDB(db_path=path).close()
    db = HistoryDB(db_path=path)  # second open: no-op migration
    try:
        assert "cwd" in _sessions_columns(db._conn)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Directory-scoped listing
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    hist = HistoryDB(db_path=str(tmp_path / "picker.db"))
    yield hist
    hist.close()


def test_create_session_records_explicit_cwd(db):
    session = db.create_session(title="Scoped", cwd="/proj/alpha")
    assert db.get_session(session["id"])["cwd"] == "/proj/alpha"


def test_create_session_defaults_cwd_to_process_dir(db, monkeypatch):
    monkeypatch.setattr("cascade.history.database.os.getcwd", lambda: "/proc/here")
    session = db.create_session(title="Default")
    assert db.get_session(session["id"])["cwd"] == "/proc/here"


def test_list_sessions_for_cwd_filters_and_counts(db):
    here = db.create_session(title="Here", provider="claude", cwd="/proj/alpha")
    db.add_message(here["id"], role="user", content="hi")
    db.add_message(here["id"], role="claude", content="yo")
    db.create_session(title="Elsewhere", cwd="/proj/beta")
    unknown = db.create_session(title="Legacy", cwd="")

    listed = db.list_sessions_for_cwd("/proj/alpha")
    ids = [s["id"] for s in listed]

    # STRICTLY this directory: neither another project's session nor a legacy
    # unknown-dir (cwd '') session leaks in -- the latter used to "ghost" into
    # every directory. Both stay reachable by id / via /history.
    assert here["id"] in ids
    assert unknown["id"] not in ids
    assert all(s["title"] != "Elsewhere" for s in listed)
    counts = {s["id"]: s["message_count"] for s in listed}
    assert counts[here["id"]] == 2


def test_list_sessions_for_cwd_newest_first(db):
    old = db.create_session(title="Old", cwd="/proj/alpha")
    new = db.create_session(title="New", cwd="/proj/alpha")
    db.add_message(new["id"], role="user", content="touch")  # bumps updated_at
    listed = db.list_sessions_for_cwd("/proj/alpha")
    assert [s["id"] for s in listed] == [new["id"], old["id"]]


# ---------------------------------------------------------------------------
# Pure formatting helpers
# ---------------------------------------------------------------------------

def test_relative_time_buckets():
    now = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)

    def mk(**kw):
        return (now - timedelta(**kw)).isoformat()

    assert _relative_time(mk(seconds=5), now) == "just now"
    assert _relative_time(mk(minutes=3), now) == "3m ago"
    assert _relative_time(mk(hours=5), now) == "5h ago"
    assert _relative_time(mk(days=2), now) == "2d ago"
    assert _relative_time(mk(days=14), now) == "2w ago"
    assert _relative_time("", now) == "unknown"
    assert _relative_time("not-a-date", now) == "unknown"


def test_format_row_includes_title_count_and_model():
    now = datetime.now(timezone.utc)
    row = _format_row(
        {
            "title": "Fix the parser",
            "message_count": 1,
            "provider": "claude",
            "model": "claude-opus-4-8",
            "updated_at": now.isoformat(),
        },
        now,
    )
    plain = row.plain
    assert "Fix the parser" in plain
    assert "1 msg" in plain and "1 msgs" not in plain  # singular
    assert "claude-opus-4-8" in plain


def test_format_row_untitled_fallback():
    now = datetime.now(timezone.utc)
    row = _format_row({"title": "", "message_count": 3, "updated_at": now.isoformat()}, now)
    assert "(untitled)" in row.plain
    assert "3 msgs" in row.plain


# ---------------------------------------------------------------------------
# Rendered modal (pilot)
# ---------------------------------------------------------------------------

class _Harness(App):
    def compose(self) -> ComposeResult:
        yield Static("base")


def _sample_sessions() -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    return [
        {"id": "amber-petal", "title": "First chat", "provider": "claude",
         "model": "claude-opus-4-8", "message_count": 4, "updated_at": now},
        {"id": "cedar-pulse", "title": "Second chat", "provider": "openai",
         "model": "gpt-5", "message_count": 9, "updated_at": now},
    ]


@pytest.mark.asyncio
async def test_picker_renders_and_selects_with_arrow_enter():
    app = _Harness()
    async with app.run_test() as pilot:
        got: list = []
        app.push_screen(SessionPickerScreen(_sample_sessions()), got.append)
        await pilot.pause()
        # If compose() raised, the screen would not be current.
        assert isinstance(app.screen, SessionPickerScreen)
        await pilot.press("down")   # move highlight to the second row
        await pilot.press("enter")
        await pilot.pause()
        assert got == ["cedar-pulse"]


@pytest.mark.asyncio
async def test_picker_j_k_navigation():
    app = _Harness()
    async with app.run_test() as pilot:
        got: list = []
        app.push_screen(SessionPickerScreen(_sample_sessions()), got.append)
        await pilot.pause()
        await pilot.press("j")   # vim-down to second row
        await pilot.press("k")   # vim-up back to first
        await pilot.press("enter")
        await pilot.pause()
        assert got == ["amber-petal"]


@pytest.mark.asyncio
async def test_picker_escape_cancels_with_none():
    app = _Harness()
    async with app.run_test() as pilot:
        got: list = []
        app.push_screen(SessionPickerScreen(_sample_sessions()), got.append)
        await pilot.pause()
        assert isinstance(app.screen, SessionPickerScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert got == [None]
