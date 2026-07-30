"""Durable run-state and genuine cancellation behavior."""

import threading
import time

import pytest

from cascade.swarm.lifecycle import (
    CancellationToken,
    RunCancelled,
    RunContext,
    RunLedger,
    RunStatus,
    TaskStatus,
    run_cancellable_shell,
)
from cascade.swarm.outcome import RunOutcome


def test_run_context_persists_task_dag_and_terminal_outcome(tmp_path):
    ledger = RunLedger(str(tmp_path / "history.db"))
    run = RunContext(
        objective="build parser then wire CLI",
        workflow="routing",
        provider="openrouter",
        session_id="session-one",
        ledger=ledger,
    )

    run.start(workflow="pipeline", model="openai/gpt-oss-120b")
    run.declare_task("parser", "build parser")
    run.declare_task("cli", "wire CLI", depends_on=("parser",))
    run.task_status("parser", "build parser", TaskStatus.RUNNING)
    run.task_status("parser", "build parser", TaskStatus.SUCCEEDED, model="mercury")
    run.task_status(
        "cli",
        "wire CLI",
        TaskStatus.SUCCEEDED,
        model="frontier",
        depends_on=("parser",),
    )
    run.add_cost(0.00125)
    run.finish(
        RunOutcome.SUCCEEDED,
        input_tokens=120,
        output_tokens=30,
        worktree_path="/tmp/review",
    )

    persisted = ledger.get_run(run.id)
    tasks = ledger.list_tasks(run.id)
    assert persisted is not None
    assert persisted["status"] == RunStatus.SUCCEEDED.value
    assert persisted["workflow"] == "pipeline"
    assert persisted["input_tokens"] == 120
    assert persisted["worktree_path"] == "/tmp/review"
    assert persisted["cost"] == pytest.approx(0.00125)
    assert [task["task_id"] for task in tasks] == ["parser", "cli"]
    assert tasks[1]["depends_on"] == ["parser"]
    assert tasks[1]["status"] == TaskStatus.SUCCEEDED.value
    ledger.close()


def test_startup_recovery_marks_open_runs_and_tasks_interrupted(tmp_path):
    path = str(tmp_path / "history.db")
    first = RunLedger(path, owner_pid=999_999_999)
    run = RunContext(objective="unfinished", workflow="fanout", ledger=first)
    run.start()
    run.declare_task("a", "task a")
    run.task_status("a", "task a", TaskStatus.RUNNING)
    first.close()

    recovered = RunLedger(path)
    assert recovered.mark_interrupted() == 1
    assert recovered.get_run(run.id)["status"] == RunStatus.INTERRUPTED.value
    task = recovered.list_tasks(run.id)[0]
    assert task["status"] == TaskStatus.CANCELLED.value
    assert "process exited" in task["error"]
    recovered.close()


def test_recovery_does_not_interrupt_a_run_owned_by_a_live_process(tmp_path):
    path = str(tmp_path / "history.db")
    live = RunLedger(path)
    run = RunContext(objective="still live", ledger=live)
    run.start()

    observer = RunLedger(path)
    assert observer.mark_interrupted() == 0
    assert observer.get_run(run.id)["status"] == RunStatus.RUNNING.value
    observer.close()
    live.close()


def test_cancel_marks_open_tasks_and_invokes_registered_cleanup(tmp_path):
    ledger = RunLedger(str(tmp_path / "history.db"))
    run = RunContext(objective="cancel me", workflow="pipeline", ledger=ledger)
    run.start()
    run.declare_task("one", "first")
    run.task_status("one", "first", TaskStatus.RUNNING)
    cleaned = []
    run.token.add_cancel_callback(lambda: cleaned.append(True))

    assert run.cancel("operator stopped it") is True
    run.finish(RunOutcome.CANCELLED, error="operator stopped it")

    assert cleaned == [True]
    assert ledger.get_run(run.id)["status"] == RunStatus.CANCELLED.value
    task = ledger.list_tasks(run.id)[0]
    assert task["status"] == TaskStatus.CANCELLED.value
    assert task["error"] == "operator stopped it"
    with pytest.raises(RunCancelled, match="operator stopped it"):
        run.checkpoint()
    ledger.close()


def test_route_metadata_and_recent_outcomes_support_receipts_and_feedback(tmp_path):
    ledger = RunLedger(str(tmp_path / "history.db"))
    for index, outcome in enumerate((
        RunOutcome.SUCCEEDED,
        RunOutcome.SUCCEEDED,
        RunOutcome.FAILED,
    )):
        run = RunContext(
            objective=f"solve {index}",
            workflow="routing",
            session_id="session-feedback",
            ledger=ledger,
        )
        run.annotate(
            route="solve",
            route_reason="focused edit",
            route_confidence=0.9,
        )
        run.start(workflow="solve")
        run.finish(outcome, metadata={"verification_kind": "test-suite"})

    recent = ledger.list_runs(session_id="session-feedback")
    assert len(recent) == 3
    assert recent[0]["metadata"]["route_reason"] == "focused edit"
    assert recent[0]["metadata"]["verification_kind"] == "test-suite"
    assert ledger.routing_summary()["solve"] == {
        "runs": 3,
        "succeeded": 2,
        "failed": 1,
    }
    ledger.close()


def test_cancellable_shell_terminates_a_running_process_promptly(tmp_path):
    token = CancellationToken()
    result = {}

    def _run() -> None:
        try:
            run_cancellable_shell(
                "python3 -c 'import time; time.sleep(30)'",
                str(tmp_path),
                60,
                token,
            )
        except Exception as exc:  # captured for assertion in the parent thread
            result["error"] = exc

    worker = threading.Thread(target=_run)
    worker.start()
    time.sleep(0.15)
    token.cancel("stop subprocess")
    worker.join(timeout=3)

    assert not worker.is_alive()
    assert isinstance(result.get("error"), RunCancelled)
