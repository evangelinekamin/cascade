"""AgentMemory: a governed shared blackboard for multi-agent runs.

A small SQLite (WAL) store that workers write structured entries to and the
orchestrator reads a curated digest from -- the coordination channel for the
swarm. Entries are typed (decision / contract / result / blocker / artifact /
fact) so the digest can surface what matters -- contracts, decisions, the latest
result per agent, open blockers -- instead of raw activity logs. Per the
multi-agent research, what coordinates agents is shared *decisions/contracts*,
not a log of who did what.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DEFAULT_DB_PATH = "~/.cache/cascade/agent_memory.db"

# Allowed entry kinds. Order is irrelevant; membership is what is validated.
KINDS: Tuple[str, ...] = ("decision", "contract", "result", "blocker", "artifact", "fact")


@dataclass(frozen=True)
class MemoryEntry:
    """One structured entry on the shared blackboard."""

    id: str
    run_id: str
    agent: str
    kind: str
    content: str
    refs: Tuple[str, ...]
    created_at: str


class AgentMemory:
    """Shared blackboard for a multi-agent run, backed by SQLite (WAL)."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:")
        else:
            path = Path(db_path or _DEFAULT_DB_PATH).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path))
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_memory (
                id          TEXT PRIMARY KEY,
                run_id      TEXT NOT NULL,
                agent       TEXT NOT NULL,
                kind        TEXT NOT NULL,
                content     TEXT NOT NULL,
                refs        TEXT NOT NULL DEFAULT '[]',
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_agent_memory_run
                ON agent_memory(run_id, id);
            """
        )
        self._conn.commit()

    def report(
        self,
        run_id: str,
        agent: str,
        kind: str,
        content: str,
        refs: Tuple[str, ...] = (),
    ) -> MemoryEntry:
        """Write a structured entry to the blackboard (worker -> store)."""
        if kind not in KINDS:
            raise ValueError(f"unknown memory kind '{kind}'; expected one of {KINDS}")
        entry = MemoryEntry(
            id=uuid.uuid4().hex[:12],
            run_id=run_id,
            agent=agent,
            kind=kind,
            content=content,
            refs=tuple(refs),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._conn.execute(
            "INSERT INTO agent_memory (id, run_id, agent, kind, content, refs, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entry.id,
                entry.run_id,
                entry.agent,
                entry.kind,
                entry.content,
                json.dumps(list(entry.refs)),
                entry.created_at,
            ),
        )
        self._conn.commit()
        return entry

    def entries(self, run_id: str, kind: Optional[str] = None) -> List[MemoryEntry]:
        """Return a run's entries in insertion order, optionally filtered by kind."""
        if kind is None:
            rows = self._conn.execute(
                "SELECT * FROM agent_memory WHERE run_id = ? ORDER BY rowid ASC",
                (run_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM agent_memory WHERE run_id = ? AND kind = ? ORDER BY rowid ASC",
                (run_id, kind),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def digest(self, run_id: str) -> str:
        """A curated view for the orchestrator.

        Contracts and decisions (all), the latest result per agent, and open
        blockers -- never raw activity logs, so the director is not drowned.
        """
        rows = self.entries(run_id)
        if not rows:
            return "No shared memory for this run yet."

        contracts = [e for e in rows if e.kind == "contract"]
        decisions = [e for e in rows if e.kind == "decision"]
        blockers = [e for e in rows if e.kind == "blocker"]
        latest_result: Dict[str, MemoryEntry] = {}
        for entry in rows:
            if entry.kind == "result":
                latest_result[entry.agent] = entry  # later rows win -> latest per agent

        sections: List[str] = []
        if contracts:
            sections.append(
                "Contracts:\n" + "\n".join(f"- {e.content}" for e in contracts)
            )
        if decisions:
            sections.append(
                "Decisions:\n" + "\n".join(f"- [{e.agent}] {e.content}" for e in decisions)
            )
        if latest_result:
            sections.append(
                "Latest result per agent:\n"
                + "\n".join(f"- [{agent}] {e.content}" for agent, e in latest_result.items())
            )
        if blockers:
            sections.append(
                "Open blockers:\n" + "\n".join(f"- [{e.agent}] {e.content}" for e in blockers)
            )
        return "\n\n".join(sections) if sections else "No shared memory for this run yet."

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            run_id=row["run_id"],
            agent=row["agent"],
            kind=row["kind"],
            content=row["content"],
            refs=tuple(json.loads(row["refs"])),
            created_at=row["created_at"],
        )

    def close(self) -> None:
        self._conn.close()
