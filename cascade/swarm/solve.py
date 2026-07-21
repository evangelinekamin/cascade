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

from ..providers.usage import Usage
from .lifecycle import (
    CancellationToken,
    RunCancelled,
    RunContext,
    TaskStatus,
    run_cancellable_shell,
)
from .outcome import RunOutcome
from .verify_loop import VerifiedWorker, VerifyAttempt, WorkerResult
from .workspace import run_agent_in_worktree
from .worktree import WorktreeManager

ProgressCallback = Optional[Callable[[str, str], None]]
TokensCallback = Optional[Callable[[int, int], None]]
CostCallback = Optional[Callable[[float], None]]

DEFAULT_TEST_CMD = "python -m pytest -x -q"

_WORKER_SYSTEM = """\
You are a coding agent working in an isolated git worktree.

Make the requested change directly in this workspace. The project's test suite
will be run to verify your work -- your goal is to make it pass. Keep the change
focused, do not ask for confirmation, and stay inside the workspace.
"""


# --- Per-model behavioral guidance for the verified-build worker ------------------
#
# Cheap "bulk" models each have their own empirically-observed failure modes. Rather
# than bloat the shared worker prompt with caveats only one model needs, a model's
# guidance is appended only on the iterations that model actually runs. Each registry
# entry pairs a lowercase substring matched against the running model id with steering
# that counters that model's dogfooded failures; ``worker_guidance_for`` concatenates
# every matching entry, so a model id can accumulate steering from more than one.

_DEEPSEEK_GUIDANCE = """\
Guidance tuned to this model's known failure modes:
- Prefer the `replace_in_file` tool for targeted edits; do NOT rewrite whole files
  with `write_file` unless necessary -- whole-file rewrites drop code and waste tokens.
- Do NOT modify shared helper functions or modules unless the task explicitly requires
  it -- incidental edits to shared code cause collateral test failures elsewhere.
- When writing ngspice/SPICE decks for analog circuits, model an active regulator as a
  behavioral CURRENT source (a transconductance, e.g. "Bout 0 vout I=..."), never a
  self-referential voltage source, or the DC operating point will not converge.
- Act using the tools; do not merely narrate what you would do.\
"""

# Substring -> guidance. Order is the concatenation order for a multi-match model id.
_WORKER_GUIDANCE: tuple[tuple[str, str], ...] = (("deepseek", _DEEPSEEK_GUIDANCE),)


def worker_guidance_for(model: str) -> str:
    """Return per-model steering to append to the worker prompt, or "" if none applies.

    Each registry entry pairs a case-insensitive substring of the running model id
    with guidance countering that model's empirically-known failure modes. The
    guidance of every matching entry is concatenated in registry order (blank-line
    separated), so a model id can accumulate steering from more than one entry.
    """
    model_id = (model or "").lower()
    matches = [text for needle, text in _WORKER_GUIDANCE if needle in model_id]
    return "\n\n".join(matches)


@dataclass(frozen=True)
class SolveResult:
    """Outcome of a verified solve run."""

    task: str
    provider: str
    passed: bool
    iterations: int
    attempts: tuple[VerifyAttempt, ...]
    worktree_path: str
    outcome: RunOutcome = RunOutcome.FAILED
    diff_stat: str = ""
    diff_excerpt: str = ""
    changed_files: tuple[str, ...] = ()
    models_used: tuple[str, ...] = ()
    providers_used: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    guardrail_fired: bool = False
    verification_kind: str = ""
    # Full re-appliable binary patch (worktree vs its baseline). Unlike the
    # clipped diff_excerpt, this is what /apply lands on the real tree.
    patch: str = ""
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


# --- Anti-proxy verification classifier -----------------------------------------
#
# A cheap model can turn the verify gate green without doing real work -- by
# pointing it at a grep, a ``test -f``, or a bare ``true``. classify_verification
# labels what the resolved verify command actually exercises so a passing run can
# be annotated when its "check" proves little. It only informs a warning; it never
# changes the pass/fail decision. Patterns use word boundaries so short tokens
# (``rg``, ``ls``, ``cat``) do not match inside unrelated words (``large``,
# ``tools``, ``concatenate``). ``bare test`` is deliberately absent from the test
# tier: ``go test``/``cargo test`` are runners, but ``test -f`` is a file check.
_TEST_RUNNER_RE = re.compile(
    r"\bpytest\b|\bunittest\b|\bjest\b|\bvitest\b|\bgo\s+test\b|\bcargo\s+test\b"
)
_SYNTACTIC_RE = re.compile(r"\bruff\b|\bflake8\b|\bmypy\b|\btsc\b|\bpy_compile\b|\beslint\b")
_PROXY_RE = re.compile(r"\bgrep\b|\brg\b|\bls\b|\bcat\b|\btest\s+-f\b|\[\s*-f\b")
_SENTINEL_RE = re.compile(r"\btouch\b|\btrue\b|(?:^|\s):(?:\s|$)")


def classify_verification(test_cmd: str) -> str:
    """Classify what the resolved verify command actually exercises.

    Returns one of:

      * ``"test"``      -- a real test runner (pytest, unittest, jest, vitest,
                           ``go test``, ``cargo test``): exercises behavior.
      * ``"syntactic"`` -- compile/lint only (ruff, flake8, mypy, tsc,
                           ``py_compile``, eslint): proves it parses, not that it
                           works.
      * ``"proxy"``     -- existence/grep checks (grep, rg, ls, cat, ``test -f``,
                           ``[ -f``): proves the check ran, nothing more.
      * ``"sentinel"``  -- a no-op that always passes (touch, true, ``:``).
      * ``"unknown"``   -- none of the above.

    A real test runner anywhere in the command wins, so ``ruff && pytest`` is
    ``"test"``. Otherwise the strongest signal present is reported.
    """
    haystack = (test_cmd or "").lower()
    if _TEST_RUNNER_RE.search(haystack):
        return "test"
    if _SYNTACTIC_RE.search(haystack):
        return "syntactic"
    if _PROXY_RE.search(haystack):
        return "proxy"
    if _SENTINEL_RE.search(haystack):
        return "sentinel"
    return "unknown"


def _annotate_verification(
    error: str,
    passed: bool,
    kind: str,
    *,
    allow_weak: bool = False,
) -> "tuple[bool, str]":
    """Reject a green proxy/sentinel gate unless explicitly permitted.

    A passing run whose verify command was only a proxy (grep/existence) or a
    sentinel (a no-op that always succeeds) may be green without exercising real
    behavior. Verified orchestration fails closed here; callers with an intentionally
    weak project gate can explicitly opt in.
    """
    if not (passed and kind in ("proxy", "sentinel")):
        return passed, error
    if allow_weak:
        note = f"warning: the passing check ({kind}) may not exercise real behavior"
        return passed, f"{error}\n{note}" if error else note
    note = (
        f"verification rejected: the configured {kind} check does not prove "
        "behavior; configure a real test command or explicitly allow a weak gate"
    )
    return False, f"{error}\n{note}" if error else note


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


def _run_tests_in(
    cmd: str,
    cwd: str,
    timeout: int,
    cancel_token: Optional[CancellationToken] = None,
) -> "tuple[str, int]":
    """Run *cmd* inside *cwd*; return (combined output, returncode)."""
    try:
        output, returncode, timed_out = run_cancellable_shell(
            cmd, cwd, timeout, cancel_token,
        )
        if timed_out:
            return f"[tests timed out after {timeout}s]", -1
        return output.strip(), returncode
    except RunCancelled:
        raise


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


def _preflight_gate(
    test_cmd: str,
    cwd: str,
    timeout: int,
    cancel_token: Optional[CancellationToken] = None,
) -> Optional[str]:
    """Confirm the verify command can execute before spending agent iterations.

    Returns a human-readable error when the command does not run at all (an
    environment/config problem the agent cannot fix); returns None when the gate
    is healthy -- whether or not its tests currently pass.
    """
    if cancel_token is None:
        output, returncode = _run_tests_in(test_cmd, cwd, timeout)
    else:
        output, returncode = _run_tests_in(test_cmd, cwd, timeout, cancel_token)
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
    worktree: str,
    baseline_ref: str,
    test_cmd: str,
    timeout: int,
    cancel_token: Optional[CancellationToken] = None,
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

    if cancel_token is None:
        _output, returncode = _run_tests_in(test_cmd, worktree, timeout)
    else:
        _output, returncode = _run_tests_in(
            test_cmd, worktree, timeout, cancel_token,
        )
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
    escalation_provider=None,
    escalation_model: Optional[str] = None,
    provider_label: str = "",
    escalation_label: str = "",
    timeout: int = 300,
    on_progress: ProgressCallback = None,
    on_tokens: TokensCallback = None,
    on_cost: CostCallback = None,
    on_tool_event=None,
    cancel_token: Optional[CancellationToken] = None,
) -> "tuple[WorkerResult, list[str], list[str]]":
    """Run the escalating verified loop for one task against an existing worktree.

    Bulk-first tiering: the first ``escalate_after`` iteration(s) use
    ``bulk_model`` on ``provider``; continued test failure escalates. When an
    ``escalation_provider`` is supplied the escalation hands the whole iteration
    to that *second provider* (running ``escalation_model``, or its own configured
    model when that is None) -- the cross-provider handoff. Without one, escalation
    stays on ``provider`` and merely lifts the model to ``frontier_model`` (the
    original within-provider behavior). ``provider_label``/``escalation_label``
    name each provider for the returned handoff record.

    Returns (WorkerResult, models used per iteration, providers used per iteration).
    """
    models_used: list[str] = []
    providers_used: list[str] = []
    state = {"iteration": 0}

    def _agent_for(iteration: int) -> "tuple[object, str, str]":
        """Resolve the (provider, model, provider label) that runs *iteration*."""
        if escalate and iteration > escalate_after:
            if escalation_provider is not None:
                model = escalation_model or escalation_provider.config.model
                return escalation_provider, model, (escalation_label or provider_label)
            return provider, frontier_model, provider_label
        return provider, bulk_model, provider_label

    def run_agent(prompt: str, path: str) -> str:
        if cancel_token is not None:
            cancel_token.checkpoint()
        state["iteration"] += 1
        agent, model, label = _agent_for(state["iteration"])
        models_used.append(model)
        providers_used.append(label)
        if on_progress:
            on_progress("editing", model)
        original_model = agent.config.model
        agent.config.model = model
        guidance = worker_guidance_for(model)
        worker_system = f"{_WORKER_SYSTEM}\n{guidance}" if guidance else _WORKER_SYSTEM
        try:
            agent_kwargs = {
                "system": worker_system,
                "max_rounds": max_rounds,
                "on_tool_event": on_tool_event,
            }
            if cancel_token is not None:
                agent_kwargs["cancel_token"] = cancel_token
            response = run_agent_in_worktree(agent, prompt, path, **agent_kwargs)
            if on_tokens is not None:
                usage = getattr(agent, "last_usage", None)
                if isinstance(usage, Usage):
                    on_tokens(usage.prompt_total, usage.output)
            cost = getattr(agent, "last_cost", None)
            if on_cost is not None and isinstance(cost, (int, float)) and cost:
                on_cost(float(cost))
            return response
        finally:
            agent.config.model = original_model

    def run_tests(path: str) -> "tuple[str, int]":
        # Restore the scaffolded gating tests (snapshotted below, once the
        # preflight gate passes) before every verification, so any worker edits
        # to them are reverted and cannot weaken the spec that grades the worker.
        _restore_files(test_snapshot)
        if on_progress:
            on_progress("verifying", f"running: {test_cmd}")
        if cancel_token is None:
            return _run_tests_in(test_cmd, path, timeout)
        return _run_tests_in(test_cmd, path, timeout, cancel_token)

    def on_attempt(attempt: VerifyAttempt) -> None:
        if on_progress:
            outcome = "passed" if attempt.passed else "failed"
            on_progress("verified", f"iteration {attempt.iteration}: tests {outcome}")

    gate_error = _preflight_gate(
        test_cmd, worktree_path, timeout, cancel_token,
    )
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
        return aborted, [], []

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
        if _minimize_blast_radius(
            worktree_path,
            guardrail_baseline,
            test_cmd,
            timeout,
            cancel_token,
        ):
            if on_progress:
                on_progress("guardrail", "reverted needless edits to shared code; suite green")
            result = replace(result, passed=True, guardrail_fired=True)

    return result, models_used, providers_used


def _resolve_escalation(app, escalate_to) -> "tuple[Optional[object], str, Optional[str]]":
    """Resolve *escalate_to* into (provider instance, provider name, model).

    *escalate_to* is either a provider name (``"glm"``) or a
    ``(provider_name, model)`` pair. The instance is looked up in
    ``app.providers``; when no explicit model is given, the provider's frontier
    model is read from ``app.config``. Returns ``(None, "", None)`` when
    *escalate_to* is falsy or names a provider that is not available -- in which
    case escalation stays within the primary provider (within-provider frontier
    lift), a graceful degradation rather than an abort.
    """
    if not escalate_to:
        return None, "", None
    if isinstance(escalate_to, (tuple, list)):
        name = escalate_to[0]
        model = escalate_to[1] if len(escalate_to) > 1 else None
    else:
        name, model = escalate_to, None
    provider = app.providers.get(name)
    if provider is None:
        return None, "", None
    if not model:
        model = app.config.get_model_for(name, fast=False)
    return provider, name, model


def run_solve(
    app,
    task: str,
    provider_name: Optional[str] = None,
    *,
    max_iterations: int = 3,
    max_rounds: int = 15,
    escalate: bool = True,
    escalate_after: int = 1,
    escalate_to: "Optional[str | tuple[str, str]]" = None,
    timeout: int = 300,
    bulk_model_override: Optional[str] = None,
    provider_preferences_override: Optional[dict] = None,
    allow_noop: bool = False,
    allow_weak_verification: bool = False,
    on_progress: ProgressCallback = None,
    on_tokens: TokensCallback = None,
    on_tool_event=None,
    cancel_token: Optional[CancellationToken] = None,
    run_context: Optional[RunContext] = None,
) -> SolveResult:
    """Run *task* to a verified diff in an isolated worktree.

    The provider edits files in a fresh git worktree; the configured test command
    runs inside that worktree each iteration, and failures are fed back until the
    tests pass or ``max_iterations`` is reached. The worktree is left in place so
    its diff can be inspected; the caller's working tree is untouched.

    When ``escalate`` is set, the first ``escalate_after`` iteration(s) run on the
    provider's fast (bulk) model and later iterations escalate -- bulk-first,
    stronger-on-failure, all in one worktree. By default escalation lifts the
    model within the same provider (fast -> frontier); pass ``escalate_to`` (a
    provider name like ``"glm"`` or a ``(provider_name, model)`` pair) to instead
    hand stalled iterations to a stronger *provider* entirely. The returned
    ``SolveResult`` records ``providers_used`` alongside ``models_used`` so the
    handoff is visible. ``bulk_model_override`` and
    ``provider_preferences_override`` let the automatic router start a tiny
    verified task on its selected fast lane without mutating global config.
    """
    provider_name = provider_name or app.config.get_default_provider()
    token = cancel_token or (run_context.token if run_context is not None else None)
    if token is not None:
        token.checkpoint()
    if run_context is not None:
        run_context.start(workflow="solve", provider=provider_name)
        run_context.declare_task("solve", task)
    provider = app.providers.get(provider_name)
    if provider is None:
        if run_context is not None:
            run_context.task_status(
                "solve",
                task,
                TaskStatus.BLOCKED,
                error=f"Provider '{provider_name}' not available",
            )
        return SolveResult(
            task=task,
            provider=provider_name,
            passed=False,
            iterations=0,
            attempts=(),
            worktree_path="",
            outcome=RunOutcome.BLOCKED,
            error=f"Provider '{provider_name}' not available",
        )
    if provider_preferences_override is not None:
        try:
            _original = provider
            provider = type(provider)(
                replace(
                    provider.config,
                    provider_preferences=dict(provider_preferences_override),
                )
            )
            # The re-instantiated provider must keep the wired gates.
            provider.hook_runner = getattr(_original, "hook_runner", None)
            provider.permission_engine = getattr(_original, "permission_engine", None)
        except Exception as exc:
            if run_context is not None:
                run_context.task_status(
                    "solve", task, TaskStatus.BLOCKED, error=str(exc),
                )
            return SolveResult(
                task=task,
                provider=provider_name,
                passed=False,
                iterations=0,
                attempts=(),
                worktree_path="",
                outcome=RunOutcome.BLOCKED,
                error=f"Could not isolate provider routing preferences: {exc}",
            )

    test_cmd = _test_command(app)
    verification_kind = classify_verification(test_cmd)
    frontier_model = app.config.get_model_for(provider_name, fast=False)
    bulk_model = (
        bulk_model_override
        or (app.config.get_bulk_model(provider_name) if escalate else frontier_model)
    )
    escalation_provider, escalation_name, escalation_model = _resolve_escalation(
        app, escalate_to
    )
    manager = WorktreeManager()
    token_totals = [0, 0]
    cost_total = [0.0]
    path = ""
    models_used: list[str] = []
    providers_used: list[str] = []

    def _accumulate_tokens(in_tokens: int, out_tokens: int) -> None:
        token_totals[0] += in_tokens
        token_totals[1] += out_tokens
        if run_context is not None:
            run_context.add_tokens(in_tokens, out_tokens)
        if on_tokens is not None:
            on_tokens(in_tokens, out_tokens)

    def _accumulate_cost(cost: float) -> None:
        cost_total[0] += cost
        if run_context is not None:
            run_context.add_cost(cost)

    try:
        if token is not None:
            token.checkpoint()
        path = manager.prepare(provider_name).path
        if run_context is not None:
            run_context.set_worktree(path)
            run_context.task_status(
                "solve", task, TaskStatus.RUNNING, worktree_path=path,
            )
        if on_progress:
            on_progress("workspace", path)
        result, models_used, providers_used = run_verified_task(
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
            escalation_provider=escalation_provider,
            escalation_model=escalation_model,
            provider_label=provider_name,
            escalation_label=escalation_name,
            timeout=timeout,
            on_progress=on_progress,
            on_tokens=_accumulate_tokens,
            on_cost=_accumulate_cost,
            on_tool_event=on_tool_event,
            cancel_token=token,
        )
        if token is not None:
            token.checkpoint()
        snapshot = manager.capture_snapshot(path)
        # Full patch for /apply -- captured only when there is something to
        # apply, so a failed/noop solve carries no patch.
        full_patch = ""
        if snapshot.changed_files:
            try:
                full_patch = manager.diff_patch(path)
            except Exception:
                full_patch = ""
        passed = result.passed
        error = result.error
        if passed and not snapshot.changed_files and not allow_noop:
            passed = False
            no_change = (
                "verification passed, but the worker produced no repository changes; "
                "use allow_noop only for an intentionally no-op task"
            )
            error = f"{error}\n{no_change}" if error else no_change
        passed, error = _annotate_verification(
            error,
            passed,
            verification_kind,
            allow_weak=allow_weak_verification,
        )
        if run_context is not None:
            run_context.task_status(
                "solve",
                task,
                TaskStatus.SUCCEEDED if passed else TaskStatus.FAILED,
                model=models_used[-1] if models_used else "",
                worktree_path=path,
                error=error,
                metadata={"iterations": result.iterations},
            )
        return SolveResult(
            task=task,
            provider=provider_name,
            passed=passed,
            iterations=result.iterations,
            attempts=result.attempts,
            worktree_path=path,
            outcome=RunOutcome.SUCCEEDED if passed else RunOutcome.FAILED,
            diff_stat=snapshot.diff_stat,
            diff_excerpt=snapshot.diff_excerpt,
            changed_files=snapshot.changed_files,
            models_used=tuple(models_used),
            providers_used=tuple(providers_used),
            input_tokens=token_totals[0],
            output_tokens=token_totals[1],
            cost=cost_total[0],
            guardrail_fired=result.guardrail_fired,
            verification_kind=verification_kind,
            patch=full_patch,
            error=error,
        )
    except RunCancelled as exc:
        snapshot = None
        if path:
            try:
                snapshot = manager.capture_snapshot(path)
            except Exception:
                pass
        if run_context is not None:
            run_context.task_status(
                "solve",
                task,
                TaskStatus.CANCELLED,
                model=models_used[-1] if models_used else "",
                worktree_path=path,
                error=str(exc),
            )
        return SolveResult(
            task=task,
            provider=provider_name,
            passed=False,
            iterations=0,
            attempts=(),
            worktree_path=path,
            outcome=RunOutcome.CANCELLED,
            diff_stat=snapshot.diff_stat if snapshot else "",
            diff_excerpt=snapshot.diff_excerpt if snapshot else "",
            changed_files=snapshot.changed_files if snapshot else (),
            models_used=tuple(models_used),
            providers_used=tuple(providers_used),
            input_tokens=token_totals[0],
            output_tokens=token_totals[1],
            cost=cost_total[0],
            verification_kind=verification_kind,
            error=str(exc),
        )
    except Exception as exc:
        if run_context is not None:
            run_context.task_status(
                "solve", task, TaskStatus.FAILED,
                worktree_path=path, error=str(exc),
            )
        return SolveResult(
            task=task,
            provider=provider_name,
            passed=False,
            iterations=0,
            attempts=(),
            worktree_path=path,
            outcome=RunOutcome.FAILED,
            input_tokens=token_totals[0],
            output_tokens=token_totals[1],
            cost=cost_total[0],
            verification_kind=verification_kind,
            error=str(exc),
        )
