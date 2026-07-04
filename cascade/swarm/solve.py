"""run_solve: the runnable verified worker.

Wires the three real callables -- worktree isolation, the worktree-scoped agent,
and a cwd-aware test runner -- into a VerifiedWorker and runs a single task to a
verified diff. Non-destructive: all work happens in an isolated git worktree, so
the caller's working tree is never touched.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Optional

from .verify_loop import VerifiedWorker, VerifyAttempt, WorkerResult
from .workspace import run_agent_in_worktree
from .worktree import WorktreeManager

ProgressCallback = Optional[Callable[[str, str], None]]
TokensCallback = Optional[Callable[[int, int], None]]

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
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


def _find_project_config(start: Optional[str] = None) -> Optional[Path]:
    """Find a project-local .cascade.yml, walking up to (and not past) the git root."""
    current = Path(start or os.getcwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".cascade.yml"
        if candidate.is_file():
            return candidate
        if (directory / ".git").exists():
            break  # stay inside the repo
    return None


def _project_verify_test() -> Optional[str]:
    """Read the verify/test command from a project-local .cascade.yml, if any.

    Accepts either the global-config shape (``workflows.verify.test``) or a
    terser top-level ``verify.test``.
    """
    path = _find_project_config()
    if path is None:
        return None
    try:
        import yaml

        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return None
    workflows = data.get("workflows", {})
    nested = workflows.get("verify", {}) if isinstance(workflows, dict) else {}
    for section in (nested, data.get("verify", {})):
        if isinstance(section, dict):
            test = section.get("test")
            if isinstance(test, str) and test.strip():
                return test.strip()
    return None


def _test_command(app) -> str:
    """Resolve the verify/test command.

    Project-local ``.cascade.yml`` wins over the global config, which wins over
    the built-in default -- so a repo can pin its own runner (e.g. ``uv run
    pytest``) without changing the user's global settings.
    """
    project = _project_verify_test()
    if project:
        return project
    try:
        verify = app.config.data.get("workflows", {}).get("verify", {})
        return verify.get("test") or DEFAULT_TEST_CMD
    except Exception:
        return DEFAULT_TEST_CMD


def _worktree_change_summary(path: str, cap: int = 1500) -> str:
    """Return a capped ``git diff --stat`` of the worker's changes so far, or ''.

    Diffs against the worktree's baseline commit (its HEAD, into which the
    manager folded any pre-existing dirt) and marks new files intent-to-add so
    freshly created files also appear. Best-effort: any git error yields ''. Fed
    into retry prompts so a re-running agent builds on prior work, not cold.
    """
    def _git(args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=path, capture_output=True, text=True, timeout=30
        )

    try:
        _git(["add", "-N", "."])
        result = _git(["diff", "--stat", "HEAD"])
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    out = (result.stdout or "").strip()
    if len(out) > cap:
        out = out[-cap:]
    return out


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


# Signatures that mean the verify command never actually ran -- an environment
# problem (missing interpreter, missing pytest, nothing collected) that a coding
# agent cannot fix by editing code. Distinct from "tests ran and reported failures".
_INFRA_FAILURE_SIGNATURES = (
    "command not found",
    "no module named pytest",
    "no module named 'pytest'",
    "no tests ran",
    "no tests were collected",
    "is not recognized as an internal or external command",
)


def _is_infra_failure(output: str, returncode: int) -> bool:
    """True when the verify command failed to RUN, not merely failed its tests."""
    if returncode in (126, 127):  # shell: not executable / command not found
        return True
    if returncode == 5:  # pytest exit code for "no tests collected"
        return True
    low = output.lower()
    return any(signature in low for signature in _INFRA_FAILURE_SIGNATURES)


def _preflight_gate(test_cmd: str, cwd: str, timeout: int) -> Optional[str]:
    """Confirm the verify command can execute before spending agent iterations.

    Returns a human-readable error when the command does not run at all (an
    environment/config problem the agent cannot fix); returns None when the gate
    is healthy -- whether or not its tests currently pass.
    """
    output, returncode = _run_tests_in(test_cmd, cwd, timeout)
    if _is_infra_failure(output, returncode):
        stripped = output.strip()
        detail = stripped.splitlines()[-1] if stripped else f"exit code {returncode}"
        return f"verify command did not run ({test_cmd!r}): {detail}"
    return None


# The verified loop OWNS the scaffolded tests -- they are the contract the worker
# is graded against -- so it protects them here rather than in the generic
# WorkspaceTools layer, which cannot know which tests are the contract.
_TEST_DIR_SEGMENTS = frozenset({"tests", "test"})


def _is_test_file(rel_path: Path) -> bool:
    """True when *rel_path* (relative to the worktree) is a gating test file.

    A file counts as a test if its basename matches ``test_*.py``, ``*_test.py``,
    or ``conftest.py``, or if any parent directory segment is ``tests`` or ``test``.
    """
    name = rel_path.name
    if fnmatch(name, "test_*.py") or fnmatch(name, "*_test.py") or name == "conftest.py":
        return True
    return any(segment in _TEST_DIR_SEGMENTS for segment in rel_path.parent.parts)


def _snapshot_test_files(worktree_path: str) -> dict[str, str]:
    """Capture path -> content for every test file currently in the worktree.

    Taken before the worker runs; restored before each verification so worker
    edits to the gating tests never count. Binary or unreadable files are skipped
    (only text that can be faithfully restored is recorded).
    """
    root = Path(worktree_path)
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _is_test_file(path.relative_to(root)):
            continue
        try:
            snapshot[str(path)] = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
    return snapshot


def _restore_files(mapping: dict[str, str]) -> None:
    """Restore each recorded path to its snapshotted content, overwriting edits.

    Only paths present in *mapping* (all of which existed at snapshot time) are
    touched; worker-created new files and the worker's implementation are left
    untouched.
    """
    for path, content in mapping.items():
        try:
            Path(path).write_text(content)
        except OSError:
            continue


def run_verified_task(
    provider,
    worktree_path: str,
    task: str,
    test_cmd: str,
    *,
    bulk_model: str,
    frontier_model: str,
    max_iterations: int = 3,
    max_rounds: int = 15,
    escalate: bool = True,
    escalate_after: int = 1,
    timeout: int = 300,
    on_progress: ProgressCallback = None,
    on_tokens: TokensCallback = None,
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
            response = run_agent_in_worktree(
                provider, prompt, path, system=_WORKER_SYSTEM, max_rounds=max_rounds
            )
            if on_tokens is not None:
                usage = getattr(provider, "last_usage", None)
                if isinstance(usage, tuple) and len(usage) == 2:
                    try:
                        on_tokens(int(usage[0]), int(usage[1]))
                    except (TypeError, ValueError):
                        pass
            return response
        finally:
            provider.config.model = original_model

    def run_tests(path: str) -> "tuple[str, int]":
        # Restore the scaffolded gating tests (snapshotted below, once the
        # preflight gate passes) before every verification, so any worker edits
        # to them are reverted and cannot weaken the spec that grades the worker.
        _restore_files(test_snapshot)
        if on_progress:
            on_progress("verifying", f"running: {test_cmd}")
        return _run_tests_in(test_cmd, path, timeout)

    def on_attempt(attempt: VerifyAttempt) -> None:
        if on_progress:
            outcome = "passed" if attempt.passed else "failed"
            on_progress("verified", f"iteration {attempt.iteration}: tests {outcome}")

    gate_error = _preflight_gate(test_cmd, worktree_path, timeout)
    if gate_error is not None:
        if on_progress:
            on_progress("aborted", gate_error)
        aborted = WorkerResult(
            task=task,
            passed=False,
            iterations=0,
            attempts=(),
            worktree_path=worktree_path,
            error=gate_error,
        )
        return aborted, []

    # Snapshot the scaffolded tests now -- after the gate proves them healthy and
    # before the worker touches anything -- so run_tests can restore them each
    # cycle, making the contract immutable to the worker.
    test_snapshot = _snapshot_test_files(worktree_path)

    worker = VerifiedWorker(
        run_agent,
        run_tests,
        lambda: worktree_path,
        max_iterations=max_iterations,
        describe_changes=_worktree_change_summary,
    )
    result = worker.run(task, on_attempt=on_attempt)
    return result, models_used


def run_solve(
    app,
    task: str,
    provider_name: Optional[str] = None,
    *,
    max_iterations: int = 3,
    max_rounds: int = 15,
    escalate: bool = True,
    escalate_after: int = 1,
    timeout: int = 300,
    on_progress: ProgressCallback = None,
    on_tokens: TokensCallback = None,
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
    token_totals = [0, 0]

    def _accumulate_tokens(in_tokens: int, out_tokens: int) -> None:
        token_totals[0] += in_tokens
        token_totals[1] += out_tokens
        if on_tokens is not None:
            on_tokens(in_tokens, out_tokens)

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
            max_rounds=max_rounds,
            escalate=escalate,
            escalate_after=escalate_after,
            timeout=timeout,
            on_progress=on_progress,
            on_tokens=_accumulate_tokens,
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
            input_tokens=token_totals[0],
            output_tokens=token_totals[1],
            error=result.error,
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
