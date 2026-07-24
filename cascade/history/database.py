"""SQLite-backed conversation history.

Schema:
  sessions - one row per conversation session
  messages - one row per user/assistant message, FK to sessions
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .wordid import generate_word_id
from ..episodes import Episode


_DEFAULT_DB_PATH = "~/.config/cascade/history.db"

# Bumped when _migrate() gains a new step. Guarded by PRAGMA user_version.
_SCHEMA_VERSION = 3


class HistoryDB:
    """Persistent conversation history stored in SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        path = Path(db_path or _DEFAULT_DB_PATH).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        self._migrate()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT '',
                provider    TEXT NOT NULL DEFAULT '',
                model       TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                metadata    TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                metadata    TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at DESC);
        """)

    def _migrate(self) -> None:
        """Apply guarded, versioned schema migrations."""
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id          TEXT NOT NULL,
                    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    timestamp   REAL NOT NULL,
                    provider    TEXT NOT NULL DEFAULT '',
                    objective   TEXT NOT NULL DEFAULT '',
                    actions     TEXT NOT NULL DEFAULT '[]',
                    outcome     TEXT NOT NULL DEFAULT '',
                    artifacts   TEXT NOT NULL DEFAULT '[]',
                    tokens      INTEGER NOT NULL DEFAULT 0,
                    turn_count  INTEGER NOT NULL DEFAULT 1,
                    source      TEXT NOT NULL DEFAULT 'live',
                    PRIMARY KEY (session_id, id)
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_session
                    ON episodes(session_id, timestamp);
            """)
            self._conn.execute("PRAGMA user_version = 1")
            self._conn.commit()
            version = 1
        if version < 2:
            # Carried context lives in its OWN table: sessions.metadata is
            # rewritten wholesale by BranchingSession._save_state on every
            # recorded message, which silently wiped anything stored there.
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS session_context (
                    session_id          TEXT PRIMARY KEY
                                        REFERENCES sessions(id) ON DELETE CASCADE,
                    compaction_summary  TEXT NOT NULL DEFAULT '',
                    compacted_through   INTEGER NOT NULL DEFAULT 0,
                    compaction_boundary TEXT NOT NULL DEFAULT '',
                    compaction_count    INTEGER NOT NULL DEFAULT 0
                );
            """)
            # Each step stamps its OWN version so the chain stays crash-safe
            # and idempotent as later steps are appended (never the moving
            # _SCHEMA_VERSION, which would mark a DB current mid-migration).
            self._conn.execute("PRAGMA user_version = 2")
            self._conn.commit()
            version = 2
        if version < 3:
            # /resume filters recent chats to the directory they happened in.
            # Added via ALTER (not _create_tables) so a fresh DB and an
            # upgraded one converge on one schema through the same path;
            # existing rows backfill to '' -- treated as unknown/any-dir.
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN cwd TEXT NOT NULL DEFAULT ''"
            )
            self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            self._conn.commit()

    def _unique_word_id(self, max_attempts: int = 20) -> str:
        """Generate a word-based ID that doesn't collide with existing sessions."""
        for _ in range(max_attempts):
            candidate = generate_word_id()
            exists = self._conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (candidate,)
            ).fetchone()
            if not exists:
                return candidate
        # Extremely unlikely fallback
        return f"{generate_word_id()}-{uuid.uuid4().hex[:4]}"

    # -- sessions --

    def create_session(
        self,
        provider: str = "",
        model: str = "",
        title: str = "",
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> dict:
        """Create a new conversation session. Returns the session dict.

        ``cwd`` records the directory the chat happened in so /resume can
        show only this project's recent sessions. Omitted -> the current
        process directory; explicit '' -> unknown (reachable from any dir).
        """
        now = datetime.now(timezone.utc).isoformat()
        resolved_cwd = os.getcwd() if cwd is None else cwd
        if session_id:
            exists = self._conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if exists:
                session_id = self._unique_word_id()
        else:
            session_id = self._unique_word_id()
        row = {
            "id": session_id,
            "title": title,
            "provider": provider,
            "model": model,
            "cwd": resolved_cwd,
            "created_at": now,
            "updated_at": now,
            "metadata": json.dumps(metadata or {}),
        }
        self._conn.execute(
            "INSERT INTO sessions (id, title, provider, model, cwd, created_at, updated_at, metadata) "
            "VALUES (:id, :title, :provider, :model, :cwd, :created_at, :updated_at, :metadata)",
            row,
        )
        self._conn.commit()
        return {**row, "metadata": metadata or {}}

    def list_sessions(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """Return recent sessions ordered by updated_at descending."""
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def list_sessions_for_cwd(self, cwd: str, limit: int = 20) -> list[dict]:
        """Recent sessions for a directory, newest first, with message_count.

        Unknown-directory sessions (cwd '') are always included so
        pre-migration rows stay reachable from any directory instead of being
        stranded. Each returned dict carries a ``message_count`` for the picker.
        """
        rows = self._conn.execute(
            "SELECT s.*, "
            "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count "
            "FROM sessions s WHERE s.cwd = ? OR s.cwd = '' "
            "ORDER BY s.updated_at DESC LIMIT ?",
            (cwd, limit),
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a single session by exact id or unique prefix match."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is not None:
            return self._row_to_session(row)
        # Try prefix match (e.g. "purple" matches "purple-banana")
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE id LIKE ? ORDER BY updated_at DESC LIMIT 2",
            (f"{session_id}%",),
        ).fetchall()
        if len(rows) == 1:
            return self._row_to_session(rows[0])
        return None

    def search_sessions(self, query: str, limit: int = 20) -> list[dict]:
        """Search sessions by title or message content."""
        rows = self._conn.execute(
            "SELECT DISTINCT s.* FROM sessions s "
            "LEFT JOIN messages m ON m.session_id = s.id "
            "WHERE s.title LIKE ? OR m.content LIKE ? "
            "ORDER BY s.updated_at DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages. Returns True if found."""
        cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def update_session_title(self, session_id: str, title: str) -> None:
        """Update a session's title."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, session_id),
        )
        self._conn.commit()

    def update_session_model(self, session_id: str, model: str) -> None:
        """Record the model most recently used in a session (for /resume)."""
        if not model:
            return
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE sessions SET model = ?, updated_at = ? WHERE id = ?",
            (model, now, session_id),
        )
        self._conn.commit()

    # -- messages --

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        token_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Add a message to a session. Returns the message dict."""
        now = datetime.now(timezone.utc).isoformat()
        msg_id = uuid.uuid4().hex[:12]
        row = {
            "id": msg_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": now,
            "token_count": token_count,
            "metadata": json.dumps(metadata or {}),
        }
        self._conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp, token_count, metadata) "
            "VALUES (:id, :session_id, :role, :content, :timestamp, :token_count, :metadata)",
            row,
        )
        # Touch the session's updated_at
        self._conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
        )
        self._conn.commit()
        return {**row, "metadata": metadata or {}}

    def get_session_messages(self, session_id: str) -> list[dict]:
        """Get all messages for a session in chronological order."""
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    # -- context carry-over (episodes + compaction summary) --

    def save_context(
        self,
        session_id: str,
        episodes: list[Episode],
        compaction_summary: str = "",
        compacted_through: int = 0,
        compaction_boundary: str = "",
        compaction_count: int = 0,
    ) -> None:
        """Snapshot the session's episode list, carried summary, and coverage.

        Replace-all semantics: the in-memory list is the source of truth
        (pruning must propagate), so the stored set always mirrors it.
        Coverage is stored two ways: ``compaction_boundary`` (content prefix
        of the newest compacted chat message -- stable across the DB
        round-trip) and ``compacted_through`` (count of compacted chat
        messages, the fallback when the boundary is ambiguous). Stored in
        session_context, never sessions.metadata -- the branching writer
        rewrites that wholesale.
        """
        self._conn.execute(
            "DELETE FROM episodes WHERE session_id = ?", (session_id,)
        )
        self._conn.executemany(
            "INSERT OR REPLACE INTO episodes "
            "(id, session_id, timestamp, provider, objective, actions, "
            " outcome, artifacts, tokens, turn_count, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    ep.id, session_id, ep.timestamp, ep.provider, ep.objective,
                    json.dumps(list(ep.actions)), ep.outcome,
                    json.dumps(list(ep.artifacts)), ep.tokens_consumed,
                    ep.raw_turn_count, ep.source,
                )
                for ep in episodes
            ],
        )
        self._conn.execute(
            "INSERT INTO session_context "
            "(session_id, compaction_summary, compacted_through, "
            " compaction_boundary, compaction_count) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "compaction_summary = excluded.compaction_summary, "
            "compacted_through = excluded.compacted_through, "
            "compaction_boundary = excluded.compaction_boundary, "
            "compaction_count = excluded.compaction_count",
            (
                session_id, compaction_summary, int(compacted_through),
                compaction_boundary, int(compaction_count),
            ),
        )
        self._conn.commit()

    def load_context(self, session_id: str) -> dict:
        """Load persisted context carry-over for a session.

        Returns {episodes, compaction_summary, compacted_through,
        compaction_boundary, compaction_count} with empty defaults.
        """
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
        episodes = [
            Episode(
                id=r["id"],
                timestamp=r["timestamp"],
                provider=r["provider"],
                objective=r["objective"],
                actions=tuple(json.loads(r["actions"])),
                outcome=r["outcome"],
                artifacts=tuple(json.loads(r["artifacts"])),
                tokens_consumed=r["tokens"],
                raw_turn_count=r["turn_count"],
                source=r["source"],
            )
            for r in rows
        ]
        row = self._conn.execute(
            "SELECT * FROM session_context WHERE session_id = ?", (session_id,)
        ).fetchone()
        out = {
            "episodes": episodes,
            "compaction_summary": "",
            "compacted_through": 0,
            "compaction_boundary": "",
            "compaction_count": 0,
        }
        if row is not None:
            out["compaction_summary"] = row["compaction_summary"] or ""
            out["compacted_through"] = int(row["compacted_through"] or 0)
            out["compaction_boundary"] = row["compaction_boundary"] or ""
            out["compaction_count"] = int(row["compaction_count"] or 0)
        return out

    # -- helpers --

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        return d

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"])
        return d

    def close(self) -> None:
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
