"""run_solve: the runnable verified worker.

Wires the three real callables -- worktree isolation, the worktree-scoped agent,
and a cwd-aware test runner -- into a VerifiedWorker and runs a single task to a
verified diff. Non-destructive: all work happens in an isolated git worktree, so
the caller's working tree is never touched.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from .verify_loop import VerifiedWorker, VerifyAttempt, WorkerResult
from .workspace import run_agent_in_worktree
from .worktree import WorktreeManager

ProgressCallback = Optional[Callable[[str, str], None]]

DEFAULT_TEST_CMD = "python -m pytest -x -q"

_WORKER_SYSTEM = """\
You are a coding agent working in an isolated git worktree.

Make the requested change directly in this workspace. The project's test suite
will be run to verify your work -- your goal is to make it pass. Keep the change
focused, do not ask for confirmation, and stay inside the workspace.
"""


@dataclass(frozen=True)
class SolveResult:
    """Outcome of a verified solve run."""

    task: str
    provider: str
    passed: bool
    iterations: int
    attempts: tuple[VerifyAttempt, ...]
    worktree_path: str
    diff_stat: str = ""
    diff_excerpt: str = ""
    changed_files: tuple[str, ...] = ()
    models_used: tuple[str, ...] = ()
    error: str = ""


def _test_command(app) -> str:
    """Resolve the verify/test command from config, falling back to a default."""
    try:
        verify = app.config.data.get("workflows", {}).get("verify", {})
        return verify.get("test") or DEFAULT_TEST_CMD
    except Exception:
        return DEFAULT_TEST_CMD


def _run_tests_in(cmd: str, cwd: str, timeout: int) -> "tuple[str, int]":
    """Run *cmd* inside *cwd*; return (combined output, returncode)."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr
        return output.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return f"[tests timed out after {timeout}s]", -1


def run_verified_task(
    provider,
    worktree_path: str,
    task: str,
    test_cmd: str,
    *,
    bulk_model: str,
    frontier_model: str,
    max_iterations: int = 3,
    escalate: bool = True,
    escalate_after: int = 1,
    timeout: int = 300,
    on_progress: ProgressCallback = None,
) -> "tuple[WorkerResult, list[str]]":
    """Run the escalating verified loop for one task against an existing worktree.

    Bulk-first model tiering: the first ``escalate_after`` iteration(s) use
    ``bulk_model`` and continued test failure escalates to ``frontier_model`` --
    all in ``worktree_path``. Returns (WorkerResult, models used per iteration).
    """
    models_used: list[str] = []
    state = {"iteration": 0}

    def _model_for(iteration: int) -> str:
        if escalate and iteration > escalate_after:
            return frontier_model
        return bulk_model

    def run_agent(prompt: str, path: str) -> str:
        state["iteration"] += 1
        model = _model_for(state["iteration"])
        models_used.append(model)
        if on_progress:
            on_progress("editing", model)
        original_model = provider.config.model
        provider.config.model = model
        try:
            return run_agent_in_worktree(provider, prompt, path, system=_WORKER_SYSTEM)
        finally:
            provider.config.model = original_model

    def run_tests(path: str) -> "tuple[str, int]":
        if on_progress:
            on_progress("verifying", f"running: {test_cmd}")
        return _run_tests_in(test_cmd, path, timeout)

    def on_attempt(attempt: VerifyAttempt) -> None:
        if on_progress:
            outcome = "passed" if attempt.passed else "failed"
            on_progress("verified", f"iteration {attempt.iteration}: tests {outcome}")

    worker = VerifiedWorker(
        run_agent, run_tests, lambda: worktree_path, max_iterations=max_iterations
    )
    result = worker.run(task, on_attempt=on_attempt)
    return result, models_used


def run_solve(
    app,
    task: str,
    provider_name: Optional[str] = None,
    *,
    max_iterations: int = 3,
    escalate: bool = True,
    escalate_after: int = 1,
    timeout: int = 300,
    on_progress: ProgressCallback = None,
) -> SolveResult:
    """Run *task* to a verified diff in an isolated worktree.

    The provider edits files in a fresh git worktree; the configured test command
    runs inside that worktree each iteration, and failures are fed back until the
    tests pass or ``max_iterations`` is reached. The worktree is left in place so
    its diff can be inspected; the caller's working tree is untouched.

    When ``escalate`` is set, the first ``escalate_after`` iteration(s) run on the
    provider's fast (bulk) model and later iterations escalate to its full
    (frontier) model -- bulk-first, frontier-on-failure, all in one worktree.
    """
    provider_name = provider_name or app.config.get_default_provider()
    provider = app.providers.get(provider_name)
    if provider is None:
        return SolveResult(
            task=task,
            provider=provider_name,
            passed=False,
            iterations=0,
            attempts=(),
            worktree_path="",
            error=f"Provider '{provider_name}' not available",
        )

    test_cmd = _test_command(app)
    frontier_model = app.config.get_model_for(provider_name, fast=False)
    bulk_model = (
        app.config.get_model_for(provider_name, fast=True) if escalate else frontier_model
    )
    manager = WorktreeManager()

    try:
        path = manager.prepare(provider_name).path
        if on_progress:
            on_progress("workspace", path)
        result, models_used = run_verified_task(
            provider,
            path,
            task,
            test_cmd,
            bulk_model=bulk_model,
            frontier_model=frontier_model,
            max_iterations=max_iterations,
            escalate=escalate,
            escalate_after=escalate_after,
            timeout=timeout,
            on_progress=on_progress,
        )
        snapshot = manager.capture_snapshot(path)
        return SolveResult(
            task=task,
            provider=provider_name,
            passed=result.passed,
            iterations=result.iterations,
            attempts=result.attempts,
            worktree_path=path,
            diff_stat=snapshot.diff_stat,
            diff_excerpt=snapshot.diff_excerpt,
            changed_files=snapshot.changed_files,
            models_used=tuple(models_used),
        )
    except Exception as exc:
        return SolveResult(
            task=task,
            provider=provider_name,
            passed=False,
            iterations=0,
            attempts=(),
            worktree_path="",
            error=str(exc),
        )
