"""run_solve: the runnable verified worker.

Wires the three real callables -- worktree isolation, the worktree-scoped agent,
and a cwd-aware test runner -- into a VerifiedWorker and runs a single task to a
verified diff. Non-destructive: all work happens in an isolated git worktree, so
the caller's working tree is never touched.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, replace
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
    guardrail_fired: bool = False
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


# --- Fix 3: blast-radius guardrail (drop a careless model's needless edits) -------
#
# A cheap worker often builds the target feature correctly but, in passing,
# needlessly rewrites or deletes pre-existing shared code it only had to call --
# collateral damage that reddens unrelated tests even though the target itself is
# done. When the worker leaves the suite red, the guardrail diffs its work against
# a pre-worker baseline, reverts only the *non-additive* hunks (removals/rewrites
# of existing code) while keeping every addition, and re-verifies once. It keeps
# the minimized tree only if that turns the suite green, so it can never make
# things worse.

_GUARDRAIL_BASELINE_MSG = "cascade-guardrail-baseline"
_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")


@dataclass(frozen=True)
class _DiffHunk:
    """One ``@@ ... @@`` hunk of a unified diff, split from its file header."""

    header: str
    body: tuple[str, ...]

    @property
    def is_additive(self) -> bool:
        """True when the hunk only adds lines (touches no pre-existing line).

        A single removal line (``-`` prefix) makes a hunk non-additive -- the
        signature of a rewrite or deletion of code that already existed. The
        file-header ``---``/``+++`` lines never reach a hunk body, so they are
        never mistaken for removals.
        """
        return not any(line.startswith("-") for line in self.body)


@dataclass(frozen=True)
class _FileDiff:
    """A single file's unified diff: its header lines plus its classified hunks."""

    header: tuple[str, ...]
    post_path: Optional[str]
    hunks: tuple[_DiffHunk, ...]


def _post_image_path(header: tuple[str, ...]) -> Optional[str]:
    """Return the worktree-relative path from a file diff's ``+++ b/<path>`` line.

    Returns None when the post-image is ``/dev/null`` (the worker deleted the
    file) -- a case the guardrail leaves alone, having no current file to revert
    into safely.
    """
    for line in header:
        if line.startswith("+++ "):
            target = line[4:].strip()
            if target == "/dev/null":
                return None
            return target[2:] if target.startswith("b/") else target
    return None


def _read_hunk_body(
    lines: list[str], start: int, match: Optional["re.Match[str]"]
) -> "tuple[tuple[str, ...], int]":
    """Read one hunk's body from *lines* at *start*; return (body, next index).

    When the ``@@`` header parsed, the body is read by its declared old/new line
    counts, so content lines that themselves look like ``@@`` or ``diff --git`` do
    not corrupt the split. When it did not parse, the body is scanned to the next
    marker as a best-effort fallback.
    """
    body: list[str] = []
    i, n = start, len(lines)
    if match is None:
        while i < n and not lines[i].startswith("@@") and not lines[i].startswith("diff --git"):
            body.append(lines[i])
            i += 1
        return tuple(body), i

    old_count = int(match.group(1)) if match.group(1) else 1
    new_count = int(match.group(2)) if match.group(2) else 1
    old_seen = new_seen = 0
    while i < n and (old_seen < old_count or new_seen < new_count):
        line = lines[i]
        if line.startswith("+"):
            new_seen += 1
        elif line.startswith("-"):
            old_seen += 1
        elif not line.startswith("\\"):  # "\ No newline at end of file" counts for neither side
            old_seen += 1
            new_seen += 1
        body.append(line)
        i += 1
    return tuple(body), i


def _parse_file_diffs(diff_text: str) -> tuple[_FileDiff, ...]:
    """Split unified ``git diff`` output into per-file diffs with classified hunks."""
    files: list[_FileDiff] = []
    lines = diff_text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        if not lines[i].startswith("diff --git"):
            i += 1
            continue
        header = [lines[i]]
        i += 1
        while i < n and not lines[i].startswith("@@") and not lines[i].startswith("diff --git"):
            header.append(lines[i])
            i += 1
        hunks: list[_DiffHunk] = []
        while i < n and lines[i].startswith("@@"):
            hunk_header = lines[i]
            body, i = _read_hunk_body(lines, i + 1, _HUNK_HEADER_RE.match(hunk_header))
            hunks.append(_DiffHunk(header=hunk_header, body=body))
        files.append(
            _FileDiff(
                header=tuple(header),
                post_path=_post_image_path(tuple(header)),
                hunks=tuple(hunks),
            )
        )
    return tuple(files)


def _nonadditive_patch(file_diffs: tuple[_FileDiff, ...]) -> "tuple[str, tuple[str, ...]]":
    """Build a patch of only the non-additive hunks, plus the paths they touch.

    For each modified file with at least one non-additive hunk, emit its header
    followed by just those hunks. Reverse-applying the result drops the worker's
    removals/rewrites of pre-existing code while leaving its additions (and any
    purely-additive files) in place. Returns ``("", ())`` when the whole diff is
    additive. Whole-file deletions are skipped -- there is no current file to
    restore into, so the guardrail leaves them for the loop to handle.
    """
    blocks: list[str] = []
    affected: list[str] = []
    for file_diff in file_diffs:
        if file_diff.post_path is None:
            continue
        non_additive = [hunk for hunk in file_diff.hunks if not hunk.is_additive]
        if not non_additive:
            continue
        section = list(file_diff.header)
        for hunk in non_additive:
            section.append(hunk.header)
            section.extend(hunk.body)
        blocks.append("\n".join(section))
        affected.append(file_diff.post_path)
    if not blocks:
        return "", ()
    return "\n".join(blocks) + "\n", tuple(affected)


def _git_in(
    worktree: str,
    args: list[str],
    input_text: Optional[str] = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Run ``git`` inside *worktree*, capturing output; callers inspect returncode."""
    return subprocess.run(
        ["git", *args],
        cwd=worktree,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _establish_guardrail_baseline(worktree: str) -> Optional[str]:
    """Commit the worktree's pre-worker state and return that commit ref, or None.

    Called after the preflight gate and before the worker's first edit, so the
    ref is a stable "before the worker touched anything" point to diff against.
    Any uncommitted state is folded into the baseline commit; an already-clean
    tree simply reuses HEAD. Any git failure yields None, which disables the
    guardrail without ever crashing the loop.
    """
    try:
        _git_in(worktree, ["add", "-A"])
        staged = _git_in(worktree, ["diff", "--cached", "--name-only"])
        if staged.returncode != 0:
            return None
        if staged.stdout.strip():
            committed = _git_in(
                worktree,
                [
                    "-c", "user.name=Cascade",
                    "-c", "user.email=cascade@local",
                    "commit", "--no-gpg-sign", "-q", "-m", _GUARDRAIL_BASELINE_MSG,
                ],
            )
            if committed.returncode != 0:
                return None
        head = _git_in(worktree, ["rev-parse", "HEAD"])
        if head.returncode != 0:
            return None
        return head.stdout.strip() or None
    except Exception:
        return None


def _minimize_blast_radius(
    worktree: str, baseline_ref: str, test_cmd: str, timeout: int
) -> bool:
    """Revert the worker's needless edits to shared code if that greens the suite.

    Diffs the worktree against *baseline_ref* (its pre-worker state) and -- only
    when some hunks are non-additive -- reverse-applies just those hunks, keeping
    every addition, then re-runs *test_cmd* once:

      * green -> the modifications were needless; the minimized tree is kept and
        True is returned (the guardrail fired).
      * red   -> the modifications were load-bearing; the worker's full changes
        are restored verbatim and False is returned.

    A pure-additive diff, an unreadable target, or any git failure short-circuits
    to False with the worker's changes left exactly as they were. The guardrail
    only ever keeps a revert that turns the suite green, so it can never make the
    suite worse.
    """
    try:
        diff = _git_in(worktree, ["diff", baseline_ref])
    except Exception:
        return False
    if diff.returncode != 0:
        return False

    patch, affected = _nonadditive_patch(_parse_file_diffs(diff.stdout))
    if not patch:
        return False  # nothing non-additive: a pure-additive build cannot regress this way

    # Snapshot the worker's current content of the files the revert will touch, so
    # a revert that fails to green can be undone without depending on git.
    saved: dict[str, str] = {}
    for rel_path in affected:
        target = str(Path(worktree) / rel_path)
        try:
            saved[target] = Path(target).read_text()
        except (OSError, UnicodeDecodeError):
            return False  # cannot guarantee restoration -> do not risk a revert

    try:
        reverted = _git_in(
            worktree, ["apply", "--reverse", "--whitespace=nowarn", "-"], input_text=patch
        )
    except Exception:
        return False
    if reverted.returncode != 0:
        return False  # git apply is atomic -- the worker's changes are untouched

    _output, returncode = _run_tests_in(test_cmd, worktree, timeout)
    if returncode == 0:
        return True  # the needless modifications are gone and the suite is green
    _restore_files(saved)  # the revert did not help -> restore the worker verbatim
    return False


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

    # Baseline the worktree at the same instant -- before the worker's first edit
    # -- so the blast-radius guardrail can diff the worker's work against a true
    # "before" state should the suite end red.
    guardrail_baseline = _establish_guardrail_baseline(worktree_path)

    worker = VerifiedWorker(
        run_agent,
        run_tests,
        lambda: worktree_path,
        max_iterations=max_iterations,
        describe_changes=_worktree_change_summary,
    )
    result = worker.run(task, on_attempt=on_attempt)

    # Blast-radius guardrail: when the worker leaves the suite red, try dropping
    # just its needless (non-additive) edits to shared code and re-verifying once.
    if not result.passed and guardrail_baseline is not None:
        _restore_files(test_snapshot)  # re-verify against the true contract
        if _minimize_blast_radius(worktree_path, guardrail_baseline, test_cmd, timeout):
            if on_progress:
                on_progress("guardrail", "reverted needless edits to shared code; suite green")
            result = replace(result, passed=True, guardrail_fired=True)

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
            guardrail_fired=result.guardrail_fired,
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
