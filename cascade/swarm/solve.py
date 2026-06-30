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

from .verify_loop import VerifiedWorker, VerifyAttempt
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


def run_solve(
    app,
    task: str,
    provider_name: Optional[str] = None,
    *,
    max_iterations: int = 3,
    timeout: int = 300,
    on_progress: ProgressCallback = None,
) -> SolveResult:
    """Run *task* to a verified diff in an isolated worktree.

    The provider edits files in a fresh git worktree; the configured test command
    runs inside that worktree each iteration, and failures are fed back until the
    tests pass or ``max_iterations`` is reached. The worktree is left in place so
    its diff can be inspected; the caller's working tree is untouched.
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
    manager = WorktreeManager()

    def prepare() -> str:
        path = manager.prepare(provider_name).path
        if on_progress:
            on_progress("workspace", path)
        return path

    def run_agent(prompt: str, path: str) -> str:
        if on_progress:
            on_progress("editing", f"{provider_name} working")
        return run_agent_in_worktree(provider, prompt, path, system=_WORKER_SYSTEM)

    def run_tests(path: str) -> "tuple[str, int]":
        if on_progress:
            on_progress("verifying", f"running: {test_cmd}")
        return _run_tests_in(test_cmd, path, timeout)

    def on_attempt(attempt: VerifyAttempt) -> None:
        if on_progress:
            outcome = "passed" if attempt.passed else "failed"
            on_progress("verified", f"iteration {attempt.iteration}: tests {outcome}")

    try:
        worker = VerifiedWorker(
            run_agent, run_tests, prepare, max_iterations=max_iterations
        )
        result = worker.run(task, on_attempt=on_attempt)
        snapshot = manager.capture_snapshot(result.worktree_path)
        return SolveResult(
            task=task,
            provider=provider_name,
            passed=result.passed,
            iterations=result.iterations,
            attempts=result.attempts,
            worktree_path=result.worktree_path,
            diff_stat=snapshot.diff_stat,
            diff_excerpt=snapshot.diff_excerpt,
            changed_files=snapshot.changed_files,
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
