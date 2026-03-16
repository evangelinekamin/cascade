"""SQLite-backed conversation history.

Schema:
  sessions - one row per conversation session
  messages - one row per user/assistant message, FK to sessions
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .wordid import generate_word_id


_DEFAULT_DB_PATH = "~/.config/cascade/history.db"


class HistoryDB:
    """Persistent conversation history stored in SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        path = Path(db_path or _DEFAULT_DB_PATH).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

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
    ) -> dict:
        """Create a new conversation session. Returns the session dict."""
        now = datetime.now(timezone.utc).isoformat()
        session_id = self._unique_word_id()
        row = {
            "id": session_id,
            "title": title,
            "provider": provider,
            "model": model,
            "created_at": now,
            "updated_at": now,
            "metadata": json.dumps(metadata or {}),
        }
        self._conn.execute(
            "INSERT INTO sessions (id, title, provider, model, created_at, updated_at, metadata) "
            "VALUES (:id, :title, :provider, :model, :created_at, :updated_at, :metadata)",
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
