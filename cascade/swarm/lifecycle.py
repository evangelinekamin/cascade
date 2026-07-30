"""Durable orchestration lifecycle and cooperative cancellation.

The UI worker object is not the run: cancelling a Textual worker only stops
Textual from awaiting it, while provider and test subprocesses can continue in
the background.  This module gives every request a stable identity, a shared
cancellation token, and a SQLite journal that survives process crashes.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .outcome import RunOutcome


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStatus(str, Enum):
    """Persisted state of a user request or orchestration run."""

    ROUTING = "routing"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class TaskStatus(str, Enum):
    """Persisted state of one task in a pipeline or fan-out DAG."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    INTEGRATED = "integrated"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


_OUTCOME_TO_STATUS = {
    RunOutcome.SUCCEEDED: RunStatus.SUCCEEDED,
    RunOutcome.PARTIAL: RunStatus.PARTIAL,
    RunOutcome.FAILED: RunStatus.FAILED,
    RunOutcome.BLOCKED: RunStatus.BLOCKED,
    RunOutcome.CANCELLED: RunStatus.CANCELLED,
}


class RunCancelled(RuntimeError):
    """Raised at a cooperative cancellation checkpoint."""


class CancellationToken:
    """Thread-safe cancellation signal shared by a run and all of its tasks."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = ""
        self._lock = threading.Lock()
        self._callbacks: dict[str, Any] = {}

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "cancelled by user") -> bool:
        """Request cancellation, returning True only for the first caller."""
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason or "cancelled"
            self._event.set()
            callbacks = list(self._callbacks.values())
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass
        return True

    def add_cancel_callback(self, callback):
        """Run *callback* on cancellation and return a function that unregisters it."""
        key = uuid.uuid4().hex
        with self._lock:
            if self._event.is_set():
                call_now = True
            else:
                self._callbacks[key] = callback
                call_now = False
        if call_now:
            callback()

        def _remove() -> None:
            with self._lock:
                self._callbacks.pop(key, None)

        return _remove

    def checkpoint(self) -> None:
        """Raise :class:`RunCancelled` if cancellation was requested."""
        if self._event.is_set():
            raise RunCancelled(self.reason or "cancelled")


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Best-effort termination of a shell command and all of its children."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=1.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait(timeout=1.0)
            except Exception:
                pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_cancellable_shell(
    command: str,
    cwd: str,
    timeout: float,
    cancel_token: Optional[CancellationToken] = None,
) -> "tuple[str, int, bool]":
    """Run a shell command with prompt cancellation and process-tree cleanup.

    Returns ``(combined_output, returncode, timed_out)``. Cancellation is not
    converted to a return code: :class:`RunCancelled` propagates after the child
    process group has been terminated.
    """
    if cancel_token is not None:
        cancel_token.checkpoint()
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=(os.name == "posix"),
    )
    started = time.monotonic()
    try:
        while True:
            if cancel_token is not None:
                cancel_token.checkpoint()
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                _terminate_process_group(proc)
                stdout, stderr = proc.communicate()
                return (stdout or "") + (stderr or ""), -1, True
            try:
                stdout, stderr = proc.communicate(timeout=min(0.1, remaining))
                if cancel_token is not None:
                    cancel_token.checkpoint()
                return (stdout or "") + (stderr or ""), int(proc.returncode or 0), False
            except subprocess.TimeoutExpired:
                continue
    except RunCancelled:
        _terminate_process_group(proc)
        raise
    except BaseException:
        _terminate_process_group(proc)
        raise


class RunLedger:
    """Thread-safe SQLite journal for runs and their task DAGs.

    It deliberately uses a connection separate from ``HistoryDB``.  Orchestration
    events are emitted by worker and fan-out threads, while conversation history
    is normally written by the Textual thread.  WAL plus a small busy timeout lets
    those two connections share one database without relaxing HistoryDB's thread
    ownership.
    """

    def __init__(self, db_path: str, owner_pid: Optional[int] = None) -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.owner_pid = int(owner_pid if owner_pid is not None else os.getpid())
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS orchestration_runs (
                    id              TEXT PRIMARY KEY,
                    session_id      TEXT NOT NULL DEFAULT '',
                    objective       TEXT NOT NULL,
                    workflow        TEXT NOT NULL,
                    provider        TEXT NOT NULL DEFAULT '',
                    model           TEXT NOT NULL DEFAULT '',
                    owner_pid       INTEGER NOT NULL DEFAULT 0,
                    status          TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    finished_at     TEXT,
                    worktree_path   TEXT NOT NULL DEFAULT '',
                    input_tokens    INTEGER NOT NULL DEFAULT 0,
                    output_tokens   INTEGER NOT NULL DEFAULT 0,
                    cost            REAL NOT NULL DEFAULT 0,
                    error           TEXT NOT NULL DEFAULT '',
                    metadata        TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_orchestration_runs_session
                    ON orchestration_runs(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_orchestration_runs_status
                    ON orchestration_runs(status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS orchestration_tasks (
                    run_id          TEXT NOT NULL
                                        REFERENCES orchestration_runs(id)
                                        ON DELETE CASCADE,
                    task_id         TEXT NOT NULL,
                    description     TEXT NOT NULL DEFAULT '',
                    status          TEXT NOT NULL,
                    provider        TEXT NOT NULL DEFAULT '',
                    model           TEXT NOT NULL DEFAULT '',
                    depends_on      TEXT NOT NULL DEFAULT '[]',
                    owned_files     TEXT NOT NULL DEFAULT '[]',
                    worktree_path   TEXT NOT NULL DEFAULT '',
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    finished_at     TEXT,
                    error           TEXT NOT NULL DEFAULT '',
                    metadata        TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (run_id, task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_orchestration_tasks_status
                    ON orchestration_tasks(run_id, status, updated_at);
                """
            )
            columns = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(orchestration_runs)")
            }
            if "owner_pid" not in columns:
                self._conn.execute(
                    "ALTER TABLE orchestration_runs "
                    "ADD COLUMN owner_pid INTEGER NOT NULL DEFAULT 0"
                )
            if "cost" not in columns:
                self._conn.execute(
                    "ALTER TABLE orchestration_runs "
                    "ADD COLUMN cost REAL NOT NULL DEFAULT 0"
                )
            self._conn.commit()

    def create_run(
        self,
        run_id: str,
        objective: str,
        workflow: str,
        *,
        session_id: str = "",
        provider: str = "",
        model: str = "",
        status: RunStatus = RunStatus.ROUTING,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO orchestration_runs (
                    id, session_id, objective, workflow, provider, model, owner_pid, status,
                    created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    objective,
                    workflow,
                    provider,
                    model,
                    self.owner_pid,
                    status.value,
                    now,
                    now,
                    json.dumps(metadata or {}),
                ),
            )
            self._conn.commit()

    def update_run(self, run_id: str, **fields: Any) -> None:
        allowed = {
            "workflow",
            "provider",
            "model",
            "status",
            "finished_at",
            "worktree_path",
            "input_tokens",
            "output_tokens",
            "cost",
            "error",
            "metadata",
        }
        values: dict[str, Any] = {}
        for key, value in fields.items():
            if key not in allowed or value is None:
                continue
            if isinstance(value, Enum):
                value = value.value
            elif key == "metadata":
                value = json.dumps(value)
            values[key] = value
        if not values:
            return
        values["updated_at"] = _now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._lock:
            self._conn.execute(
                f"UPDATE orchestration_runs SET {assignments} WHERE id = ?",
                (*values.values(), run_id),
            )
            self._conn.commit()

    def upsert_task(
        self,
        run_id: str,
        task_id: str,
        description: str,
        *,
        status: TaskStatus = TaskStatus.PENDING,
        provider: str = "",
        model: str = "",
        depends_on: tuple[str, ...] = (),
        owned_files: tuple[str, ...] = (),
        worktree_path: str = "",
        error: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        now = _now()
        finished_at = now if status in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.INTEGRATED,
            TaskStatus.CANCELLED,
            TaskStatus.SKIPPED,
        } else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO orchestration_tasks (
                    run_id, task_id, description, status, provider, model,
                    depends_on, owned_files, worktree_path, created_at, updated_at,
                    finished_at, error, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, task_id) DO UPDATE SET
                    description = excluded.description,
                    status = excluded.status,
                    provider = excluded.provider,
                    model = excluded.model,
                    depends_on = excluded.depends_on,
                    owned_files = excluded.owned_files,
                    worktree_path = excluded.worktree_path,
                    updated_at = excluded.updated_at,
                    finished_at = excluded.finished_at,
                    error = excluded.error,
                    metadata = excluded.metadata
                """,
                (
                    run_id,
                    task_id,
                    description,
                    status.value,
                    provider,
                    model,
                    json.dumps(depends_on),
                    json.dumps(owned_files),
                    worktree_path,
                    now,
                    now,
                    finished_at,
                    error,
                    json.dumps(metadata or {}),
                ),
            )
            self._conn.commit()

    def cancel_open_tasks(self, run_id: str, reason: str) -> None:
        now = _now()
        open_states = (TaskStatus.PENDING.value, TaskStatus.RUNNING.value)
        with self._lock:
            self._conn.execute(
                """
                UPDATE orchestration_tasks
                SET status = ?, updated_at = ?, finished_at = ?, error = ?
                WHERE run_id = ? AND status IN (?, ?)
                """,
                (
                    TaskStatus.CANCELLED.value,
                    now,
                    now,
                    reason,
                    run_id,
                    *open_states,
                ),
            )
            self._conn.commit()

    def add_tokens(self, run_id: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE orchestration_runs
                SET input_tokens = input_tokens + ?,
                    output_tokens = output_tokens + ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (input_tokens, output_tokens, _now(), run_id),
            )
            self._conn.commit()

    def add_cost(self, run_id: str, cost: float) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE orchestration_runs
                SET cost = cost + ?, updated_at = ?
                WHERE id = ?
                """,
                (float(cost), _now(), run_id),
            )
            self._conn.commit()

    def mark_interrupted(self) -> int:
        """Mark open runs whose owning process no longer exists as interrupted."""
        now = _now()
        open_states = (
            RunStatus.ROUTING.value,
            RunStatus.RUNNING.value,
            RunStatus.CANCELLING.value,
        )
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, owner_pid FROM orchestration_runs
                WHERE status IN (?, ?, ?)
                """,
                open_states,
            ).fetchall()
            stale_ids = [
                str(row["id"])
                for row in rows
                if not self._pid_is_alive(int(row["owner_pid"] or 0))
            ]
            if not stale_ids:
                return 0
            placeholders = ", ".join("?" for _ in stale_ids)
            cur = self._conn.execute(
                f"""
                UPDATE orchestration_runs
                SET status = ?, updated_at = ?, finished_at = ?,
                    error = CASE WHEN error = ''
                        THEN 'process exited before the run reached a terminal state'
                        ELSE error END
                WHERE id IN ({placeholders})
                """,
                (RunStatus.INTERRUPTED.value, now, now, *stale_ids),
            )
            self._conn.execute(
                f"""
                UPDATE orchestration_tasks
                SET status = ?, updated_at = ?, finished_at = ?,
                    error = CASE WHEN error = ''
                        THEN 'process exited before the task reached a terminal state'
                        ELSE error END
                WHERE status IN (?, ?)
                  AND run_id IN ({placeholders})
                """,
                (
                    TaskStatus.CANCELLED.value,
                    now,
                    now,
                    TaskStatus.PENDING.value,
                    TaskStatus.RUNNING.value,
                    *stale_ids,
                ),
            )
            self._conn.commit()
            return cur.rowcount

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM orchestration_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._decode_run(row) if row is not None else None

    def list_runs(
        self,
        *,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return newest runs, optionally restricted to one session."""
        safe_limit = max(1, min(int(limit), 1000))
        with self._lock:
            if session_id is None:
                rows = self._conn.execute(
                    """
                    SELECT * FROM orchestration_runs
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT * FROM orchestration_runs
                    WHERE session_id = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (session_id, safe_limit),
                ).fetchall()
        return [self._decode_run(row) for row in rows]

    def merge_metadata(self, run_id: str, patch: dict[str, Any]) -> None:
        """Merge JSON metadata without discarding fields written by another phase."""
        if not patch:
            return
        with self._lock:
            row = self._conn.execute(
                "SELECT metadata FROM orchestration_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return
            try:
                metadata = json.loads(row["metadata"])
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.update(patch)
            self._conn.execute(
                """
                UPDATE orchestration_runs
                SET metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(metadata), _now(), run_id),
            )
            self._conn.commit()

    def routing_summary(self, *, limit: int = 100) -> dict[str, dict[str, int]]:
        """Aggregate recent terminal outcomes for weak routing tie-breakers."""
        runs = self.list_runs(limit=limit)
        summary: dict[str, dict[str, int]] = {}
        terminal = {
            RunStatus.SUCCEEDED.value,
            RunStatus.PARTIAL.value,
            RunStatus.FAILED.value,
            RunStatus.BLOCKED.value,
            RunStatus.CANCELLED.value,
            RunStatus.INTERRUPTED.value,
        }
        for run in runs:
            workflow = str(run.get("workflow") or "")
            status = str(run.get("status") or "")
            if not workflow or workflow == "routing" or status not in terminal:
                continue
            bucket = summary.setdefault(
                workflow, {"runs": 0, "succeeded": 0, "failed": 0}
            )
            bucket["runs"] += 1
            if status == RunStatus.SUCCEEDED.value:
                bucket["succeeded"] += 1
            elif status != RunStatus.PARTIAL.value:
                bucket["failed"] += 1
        return summary

    def list_tasks(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM orchestration_tasks
                WHERE run_id = ? ORDER BY created_at, task_id
                """,
                (run_id,),
            ).fetchall()
        return [self._decode_task(row) for row in rows]

    @staticmethod
    def _decode_run(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["metadata"] = json.loads(data["metadata"])
        return data

    @staticmethod
    def _decode_task(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for key in ("depends_on", "owned_files", "metadata"):
            data[key] = json.loads(data[key])
        return data

    def close(self) -> None:
        with self._lock:
            self._conn.close()


@dataclass
class RunContext:
    """Live handle connecting cancellation, durable status, and task events."""

    objective: str
    workflow: str = "routing"
    provider: str = ""
    model: str = ""
    session_id: str = ""
    ledger: Optional[RunLedger] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    token: CancellationToken = field(default_factory=CancellationToken)
    _terminal: bool = field(default=False, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.ledger is not None:
            self.ledger.create_run(
                self.id,
                self.objective,
                self.workflow,
                session_id=self.session_id,
                provider=self.provider,
                model=self.model,
            )

    @property
    def cancelled(self) -> bool:
        return self.token.cancelled

    def checkpoint(self) -> None:
        self.token.checkpoint()

    def start(
        self,
        *,
        workflow: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        worktree_path: Optional[str] = None,
    ) -> None:
        with self._lock:
            if self._terminal:
                return
            if workflow is not None:
                self.workflow = workflow
            if provider is not None:
                self.provider = provider
            if model is not None:
                self.model = model
            if self.ledger is not None:
                self.ledger.update_run(
                    self.id,
                    workflow=self.workflow,
                    provider=self.provider,
                    model=self.model,
                    worktree_path=worktree_path,
                    status=RunStatus.RUNNING,
                )

    def set_worktree(self, path: str) -> None:
        if self.ledger is not None:
            self.ledger.update_run(self.id, worktree_path=path)

    def declare_task(
        self,
        task_id: str,
        description: str,
        *,
        depends_on: tuple[str, ...] = (),
        owned_files: tuple[str, ...] = (),
        worktree_path: str = "",
    ) -> None:
        if self.ledger is not None:
            self.ledger.upsert_task(
                self.id,
                task_id,
                description,
                provider=self.provider,
                depends_on=depends_on,
                owned_files=owned_files,
                worktree_path=worktree_path,
            )

    def task_status(
        self,
        task_id: str,
        description: str,
        status: TaskStatus,
        *,
        model: str = "",
        depends_on: tuple[str, ...] = (),
        owned_files: tuple[str, ...] = (),
        worktree_path: str = "",
        error: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.ledger is not None:
            self.ledger.upsert_task(
                self.id,
                task_id,
                description,
                status=status,
                provider=self.provider,
                model=model,
                depends_on=depends_on,
                owned_files=owned_files,
                worktree_path=worktree_path,
                error=error,
                metadata=metadata,
            )

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        if self.ledger is not None and (input_tokens or output_tokens):
            self.ledger.add_tokens(self.id, input_tokens, output_tokens)

    def add_cost(self, cost: float) -> None:
        if self.ledger is not None and cost:
            self.ledger.add_cost(self.id, cost)

    def annotate(self, **metadata: Any) -> None:
        """Attach explainability/receipt fields to this run."""
        if self.ledger is not None:
            self.ledger.merge_metadata(
                self.id,
                {key: value for key, value in metadata.items() if value is not None},
            )

    def cancel(self, reason: str = "cancelled by user") -> bool:
        first = self.token.cancel(reason)
        with self._lock:
            if first and not self._terminal and self.ledger is not None:
                self.ledger.update_run(
                    self.id,
                    status=RunStatus.CANCELLING,
                    error=self.token.reason,
                )
                self.ledger.cancel_open_tasks(self.id, self.token.reason)
        return first

    def finish(
        self,
        outcome: RunOutcome,
        *,
        error: str = "",
        worktree_path: str = "",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            if self._terminal:
                return
            self._terminal = True
            if self.ledger is not None:
                fields: dict[str, Any] = {
                    "status": _OUTCOME_TO_STATUS[outcome],
                    "finished_at": _now(),
                    "error": error,
                }
                if metadata:
                    self.ledger.merge_metadata(self.id, metadata)
                if worktree_path:
                    fields["worktree_path"] = worktree_path
                if input_tokens is not None:
                    fields["input_tokens"] = input_tokens
                if output_tokens is not None:
                    fields["output_tokens"] = output_tokens
                if cost is not None:
                    fields["cost"] = cost
                self.ledger.update_run(self.id, **fields)
                if outcome == RunOutcome.CANCELLED:
                    self.ledger.cancel_open_tasks(
                        self.id, error or self.token.reason or "cancelled"
                    )
