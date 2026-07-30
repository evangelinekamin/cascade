"""/solve creates a ledgered RunContext so crashes leave a recoverable trace."""

import tempfile
from pathlib import Path

from cascade.swarm.lifecycle import RunContext, RunLedger
from cascade.swarm.outcome import RunOutcome


def test_run_context_creates_and_finishes_a_ledger_row():
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "runs.db")
        ledger = RunLedger(db)
        ctx = RunContext(
            objective="fix the parser",
            workflow="solve",
            provider="openrouter",
            session_id="sess-1",
            ledger=ledger,
        )
        ctx.start(workflow="solve", provider="openrouter")
        # A crash here would leave the row non-terminal -> marked interrupted.
        ctx.finish(RunOutcome.SUCCEEDED)

        # A fresh ledger over the same DB marks nothing interrupted (row is
        # terminal), proving finish() closed it.
        ledger.close()
        ledger2 = RunLedger(db)
        recovered = ledger2.mark_interrupted()
        assert recovered == 0
        ledger2.close()


def test_run_row_is_created_with_objective_and_workflow():
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "runs.db")
        ledger = RunLedger(db)
        ctx = RunContext(
            objective="fix the parser", workflow="solve", provider="openrouter",
            session_id="sess-1", ledger=ledger,
        )
        ctx.start(workflow="solve", provider="openrouter")
        # The row exists in the ledger (a crash would leave it recoverable by
        # mark_interrupted once this process's PID is gone).
        row = ledger._conn.execute(
            "SELECT objective, workflow FROM orchestration_runs WHERE id = ?",
            (ctx.id,),
        ).fetchone()
        assert row is not None
        assert row[0] == "fix the parser"
        assert row[1] == "solve"
        ctx.finish(RunOutcome.FAILED)
        ledger.close()
