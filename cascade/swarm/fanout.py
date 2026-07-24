"""run_fanout: the parallel verified orchestrator (build-mode north star).

A frontier "director" decomposes an objective into INDEPENDENT subtasks that each
own a distinct set of files; every subtask runs as a verified worker (with the
same bulk->frontier / cross-provider escalation as /solve) in its OWN isolated
worktree, and the passing subtasks' diffs are replayed onto one integration
worktree, which is then re-verified as a whole.

Where /pipeline is sequential -- one evolving worktree, each step building on the
last -- this is a fan-out: independent subtasks that don't see each other, then a
merge. Non-destructive: the caller's working tree is never touched.

Subtasks run concurrently on independent provider instances. Director-provided
labels are never used as filesystem paths, and declared file ownership is checked
against each worker's actual diff before integration.
"""

from __future__ import annotations

import json
import inspect
import re
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from threading import Lock
from typing import Callable, List, Optional

from .lifecycle import CancellationToken, RunCancelled, RunContext, TaskStatus
from ..providers.usage import Usage
from .outcome import RunOutcome
from .solve import (
    _annotate_verification,
    _resolve_escalation,
    _run_tests_in,
    _test_command,
    classify_verification,
    run_verified_task,
)
from .worktree import WorktreeManager

ProgressCallback = Optional[Callable[[str, str], None]]
TokensCallback = Optional[Callable[[int, int], None]]

_MAX_SUBTASKS = 6
# Concurrent subtask builds. Each is a heavy verified /solve loop, so cap it.
_MAX_PARALLEL = 4

_DIRECTOR_SYSTEM = """\
You are a software director planning a PARALLEL build. Decompose the objective
into INDEPENDENT subtasks that can each be implemented and tested on their own,
in any order, without depending on another subtask's changes. Critically, each
subtask should own a DISTINCT set of files -- two subtasks must not edit the same
file, or their work will collide when merged.

Respond with JSON only:
{
  "subtasks": [
    {"id": "sub_1", "description": "short summary", "prompt": "the full self-contained instruction", "files": ["path/it/owns.py"], "required": true}
  ]
}

Keep it to 2-6 subtasks. If the objective is not cleanly separable into
independent pieces, return a single subtask covering the whole thing.
"""


@dataclass(frozen=True)
class FanoutTask:
    """One independent subtask the director planned."""

    id: str
    description: str
    prompt: str
    files: tuple[str, ...] = ()
    required: bool = True


@dataclass(frozen=True)
class FanoutSub:
    """The outcome of one subtask and whether it merged into the integration."""

    id: str
    description: str
    passed: bool  # the subtask's own tests passed in its worktree
    integrated: bool  # its verified diff applied cleanly into the integration worktree
    models_used: tuple[str, ...] = ()
    required: bool = True
    changed_files: tuple[str, ...] = ()
    ownership_violations: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class FanoutResult:
    """Outcome of a full parallel fan-out."""

    objective: str
    provider: str
    worktree_path: str  # the integration worktree
    subs: tuple[FanoutSub, ...]
    passed: bool  # the integrated whole passes the test suite
    outcome: RunOutcome = RunOutcome.FAILED
    verification_kind: str = ""
    diff_stat: str = ""
    diff_excerpt: str = ""
    changed_files: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    # Per-provider breakdown, same shape as SolveResult's: an escalated subtask's
    # tokens and cost belong to the model that incurred them, not to the base
    # provider. (provider, input_tokens, output_tokens) and (provider, cost).
    tokens_by_provider: tuple[tuple[str, int, int], ...] = ()
    cost_by_provider: tuple[tuple[str, float], ...] = ()
    error: str = ""


def _parse_subtasks(objective: str, response: str) -> List[FanoutTask]:
    """Parse the director's JSON into FanoutTasks, with a whole-objective fallback."""
    fallback = [FanoutTask(id="sub_1", description=objective, prompt=objective)]
    match = re.search(r"\{[\s\S]*\}", response or "")
    if not match:
        return fallback
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return fallback

    tasks: List[FanoutTask] = []
    for index, raw in enumerate(data.get("subtasks", []), start=1):
        if not isinstance(raw, dict):
            continue
        description = str(raw.get("description", "")).strip()
        prompt = str(raw.get("prompt", "")).strip() or description
        if not prompt:
            continue
        files = tuple(str(f) for f in raw.get("files", []) if isinstance(f, str))
        tasks.append(
            FanoutTask(
                id=str(raw.get("id", f"sub_{index}")),
                description=description or prompt,
                prompt=prompt,
                files=files,
                required=raw.get("required", True) is not False,
            )
        )
    return (tasks or fallback)[:_MAX_SUBTASKS]


_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _normalize_owned_path(raw: str) -> str:
    """Return a safe repo-relative POSIX path or raise ValueError."""
    value = raw.strip().replace("\\", "/")
    parts = value.split("/")
    if (
        not value
        or value.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
        or (parts and ":" in parts[0])
    ):
        raise ValueError(f"unsafe owned path: {raw!r}")
    return str(PurePosixPath(value))


def _paths_overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _validate_subtasks(tasks: List[FanoutTask]) -> "tuple[List[FanoutTask], str]":
    """Validate director output before creating worktrees or starting workers."""
    if not tasks:
        return [], "director produced no subtasks"

    normalized: List[FanoutTask] = []
    seen_ids: set[str] = set()
    owners: list[tuple[str, str]] = []
    for task in tasks:
        task_id = task.id.strip()
        if not _TASK_ID_RE.fullmatch(task_id):
            return [], f"director produced unsafe task id: {task.id!r}"
        if task_id in seen_ids:
            return [], f"director produced duplicate task id: {task_id}"
        seen_ids.add(task_id)
        try:
            files = tuple(dict.fromkeys(_normalize_owned_path(path) for path in task.files))
        except ValueError as exc:
            return [], str(exc)
        if len(tasks) > 1 and not files:
            return [], f"parallel subtask {task_id} has no declared file ownership"
        for path in files:
            for other_id, other_path in owners:
                if _paths_overlap(path, other_path):
                    return [], (
                        f"file ownership overlaps: {task_id}:{path} and "
                        f"{other_id}:{other_path}"
                    )
            owners.append((task_id, path))
        normalized.append(replace(task, id=task_id, files=files))
    return normalized, ""


def _ownership_violations(changed: tuple[str, ...], owned: tuple[str, ...]) -> tuple[str, ...]:
    """Changed paths not covered by a task's declared ownership."""
    if not owned:
        return ()
    violations = []
    for raw in changed:
        try:
            path = _normalize_owned_path(raw)
        except ValueError:
            violations.append(raw)
            continue
        if not any(path == root or path.startswith(root + "/") for root in owned):
            violations.append(path)
    return tuple(violations)


def plan_subtasks(
    app,
    objective: str,
    director,
    director_model: str,
    on_progress: ProgressCallback = None,
    on_tokens: TokensCallback = None,
    on_cost: Optional[Callable[[float], None]] = None,
    cancel_token: Optional[CancellationToken] = None,
) -> List[FanoutTask]:
    """Ask the *director* (a frontier model) to split the objective into subtasks."""
    if director is None:
        return [FanoutTask(id="sub_1", description=objective, prompt=objective)]
    if on_progress:
        on_progress("planning", f"director ({director_model}) decomposing into subtasks")

    original_model = director.config.model
    director.config.model = director_model
    try:
        if cancel_token is not None:
            cancel_token.checkpoint()
        cancellation_scope = getattr(director, "cancellation_scope", None)
        scope = (
            cancellation_scope(cancel_token)
            if callable(cancellation_scope) and cancel_token is not None
            else nullcontext()
        )
        with scope:
            response = director.ask_single(
                f"Decompose this objective into independent parallel subtasks:\n\n{objective}",
                system=_DIRECTOR_SYSTEM,
            )
        if cancel_token is not None:
            cancel_token.checkpoint()
        usage = getattr(director, "last_usage", None)
        if on_tokens is not None and isinstance(usage, Usage):
            on_tokens(usage.prompt_total, usage.output)
        cost = getattr(director, "last_cost", None)
        if on_cost is not None and isinstance(cost, (int, float)) and cost:
            on_cost(float(cost))
    except RunCancelled:
        raise
    except Exception:
        return [FanoutTask(id="sub_1", description=objective, prompt=objective)]
    finally:
        director.config.model = original_model

    return _parse_subtasks(objective, response)


def _clone_provider(provider):
    """A fresh provider instance with a COPIED config, safe to mutate in one thread.

    Subtasks build in parallel and run_verified_task swaps ``config.model`` in place,
    so they must not share a provider -- or even a config object. Best-effort: if the
    config is not a dataclass (e.g. a test double), deep-copy the instance. Sharing
    a mutable provider across concurrent workers is never a safe fallback.
    """
    if provider is None:
        return None
    try:
        clone = type(provider)(replace(provider.config))
    except Exception:
        clone = deepcopy(provider)
        if clone is provider:
            raise RuntimeError("provider could not be isolated for a parallel worker")
    # Clones must keep the hook + permission gates the original was wired with.
    clone.hook_runner = getattr(provider, "hook_runner", None)
    clone.permission_engine = getattr(provider, "permission_engine", None)
    return clone


def run_fanout(
    app,
    objective: str,
    provider_name: Optional[str] = None,
    *,
    max_iterations: int = 3,
    escalate: bool = True,
    escalate_after: int = 1,
    timeout: int = 300,
    allow_weak_verification: bool = False,
    on_progress: ProgressCallback = None,
    on_tokens: TokensCallback = None,
    cancel_token: Optional[CancellationToken] = None,
    run_context: Optional[RunContext] = None,
) -> FanoutResult:
    """Decompose *objective* into independent subtasks, build each verified in its
    own worktree, merge the passing diffs, and re-verify the whole.
    """
    provider_name = provider_name or app.config.get_default_provider()
    token = cancel_token or (run_context.token if run_context is not None else None)
    if token is not None:
        token.checkpoint()
    if run_context is not None:
        run_context.start(workflow="fanout", provider=provider_name)
    provider = app.providers.get(provider_name)
    if provider is None:
        return FanoutResult(
            objective=objective,
            provider=provider_name,
            worktree_path="",
            subs=(),
            passed=False,
            outcome=RunOutcome.BLOCKED,
            error=f"Provider '{provider_name}' not available",
        )

    frontier_model = app.config.get_model_for(provider_name, fast=False)
    bulk_model = app.config.get_bulk_model(provider_name) if escalate else frontier_model
    escalation_provider, escalation_name, escalation_model = _resolve_escalation(
        app, app.config.get_escalation_target(provider_name)
    )
    # Frontier directs bulk: decompose on the stronger escalation model (e.g. Opus)
    # when one is configured, else the primary provider's own frontier model.
    if escalation_provider is not None:
        director = escalation_provider
        director_name = escalation_name
        director_model = escalation_model or escalation_provider.config.model
    else:
        director = provider
        director_name = provider_name
        director_model = frontier_model
    token_totals = [0, 0]
    cost_total = [0.0]
    tokens_by_provider: dict[str, list[int]] = {}
    cost_by_provider: dict[str, float] = {}
    token_lock = Lock()

    def _accumulate_tokens(input_tokens: int, output_tokens: int) -> None:
        with token_lock:
            token_totals[0] += input_tokens
            token_totals[1] += output_tokens
        if run_context is not None:
            run_context.add_tokens(input_tokens, output_tokens)
        if on_tokens is not None:
            on_tokens(input_tokens, output_tokens)

    def _accumulate_cost(cost: float) -> None:
        with token_lock:
            cost_total[0] += cost
        if run_context is not None:
            run_context.add_cost(cost)

    def _credit_tokens(label: str, input_tokens: int, output_tokens: int) -> None:
        with token_lock:
            bucket = tokens_by_provider.setdefault(label, [0, 0])
            bucket[0] += input_tokens
            bucket[1] += output_tokens

    def _credit_cost(label: str, cost: float) -> None:
        with token_lock:
            cost_by_provider[label] = cost_by_provider.get(label, 0.0) + cost

    def _token_breakdown() -> "tuple[tuple[str, int, int], ...]":
        return tuple(
            (label, tokens[0], tokens[1]) for label, tokens in tokens_by_provider.items()
        )

    def _credit_subtask(attribution: list) -> None:
        """Fold one subtask's per-provider breakdown into the run-wide one.

        Called from the worker threads, so every credit takes the lock; the two
        loops are separate acquisitions rather than one held across both.
        """
        for label, sub_in, sub_out in (attribution[0] if attribution else ()):
            _credit_tokens(label, sub_in, sub_out)
        for label, sub_cost in (attribution[1] if len(attribution) > 1 else ()):
            _credit_cost(label, sub_cost)

    def _plan_tokens(input_tokens: int, output_tokens: int) -> None:
        """Planning runs on the director's provider, not on a subtask worker's."""
        _credit_tokens(director_name, input_tokens, output_tokens)
        _accumulate_tokens(input_tokens, output_tokens)

    def _plan_cost(cost: float) -> None:
        _credit_cost(director_name, cost)
        _accumulate_cost(cost)

    plan_kwargs = {
        "on_progress": on_progress,
        "on_tokens": _plan_tokens,
    }
    if token is not None:
        plan_kwargs["cancel_token"] = token
    if "on_cost" in inspect.signature(plan_subtasks).parameters:
        plan_kwargs["on_cost"] = _plan_cost
    tasks = plan_subtasks(
        app, objective, director, director_model, **plan_kwargs,
    )
    tasks, plan_error = _validate_subtasks(tasks)
    if plan_error:
        return FanoutResult(
            objective=objective,
            provider=provider_name,
            worktree_path="",
            subs=(),
            passed=False,
            outcome=RunOutcome.BLOCKED,
            input_tokens=token_totals[0],
            output_tokens=token_totals[1],
            tokens_by_provider=_token_breakdown(),
            cost_by_provider=tuple(cost_by_provider.items()),
            error=f"invalid parallel plan: {plan_error}",
        )
    if run_context is not None:
        for task in tasks:
            run_context.declare_task(
                task.id,
                task.description,
                owned_files=task.files,
            )
    test_cmd = _test_command(app)
    verification_kind = classify_verification(test_cmd)
    manager = WorktreeManager()
    integration = None
    subs: List[FanoutSub] = []

    try:
        if token is not None:
            token.checkpoint()
        # Preflight: a fan-out only makes sense on a green base. A red base forces
        # every subtask to also fix the same pre-existing failures, and those fixes
        # then collide on merge -- so check once up front and bail with a clear
        # message rather than producing a confusing pile of conflicts.
        base = manager.prepare("_base")
        if on_progress:
            on_progress("preflight", f"checking the base is green: {test_cmd}")
        if token is None:
            _base_output, base_rc = _run_tests_in(test_cmd, base.path, timeout)
        else:
            _base_output, base_rc = _run_tests_in(
                test_cmd, base.path, timeout, token,
            )
        if base_rc != 0:
            manager.cleanup()
            return FanoutResult(
                objective=objective,
                provider=provider_name,
                worktree_path="",
                subs=(),
                passed=False,
                outcome=RunOutcome.BLOCKED,
                input_tokens=token_totals[0],
                output_tokens=token_totals[1],
                tokens_by_provider=_token_breakdown(),
                cost_by_provider=tuple(cost_by_provider.items()),
                error=(
                    "base test suite is not green -- fix it before fanning out "
                    "(a red base makes every subtask fix the same failures and "
                    "collide on merge)"
                ),
            )

        # 1. Build each independent subtask verified, IN PARALLEL -- each in its own
        #    pre-prepared worktree on its own provider clone (run_verified_task swaps
        #    config.model in place, so subtasks must not share a provider or config).
        #    Worktrees are prepared sequentially up front (the manager is not
        #    thread-safe); only the heavy build loop runs concurrently.
        prepared = {
            task.id: manager.prepare(f"task_{index:02d}")
            for index, task in enumerate(tasks, start=1)
        }
        if run_context is not None:
            for task in tasks:
                run_context.task_status(
                    task.id,
                    task.description,
                    TaskStatus.PENDING,
                    owned_files=task.files,
                    worktree_path=prepared[task.id].path,
                )

        def _build(task):
            if token is not None:
                token.checkpoint()
            if run_context is not None:
                run_context.task_status(
                    task.id,
                    task.description,
                    TaskStatus.RUNNING,
                    owned_files=task.files,
                    worktree_path=prepared[task.id].path,
                )
            if on_progress:
                on_progress("subtask", f"{task.id}: {task.description}")
            try:
                result, models_used, _providers, *attribution = run_verified_task(
                    _clone_provider(provider),
                    prepared[task.id].path,
                    task.prompt,
                    test_cmd,
                    bulk_model=bulk_model,
                    frontier_model=frontier_model,
                    max_iterations=max_iterations,
                    escalate=escalate,
                    escalate_after=escalate_after,
                    escalation_provider=_clone_provider(escalation_provider),
                    escalation_model=escalation_model,
                    provider_label=provider_name,
                    escalation_label=escalation_name,
                    timeout=timeout,
                    on_progress=on_progress,
                    on_tokens=_accumulate_tokens,
                    on_cost=_accumulate_cost,
                    cancel_token=token,
                )
                _credit_subtask(attribution)
                if token is not None:
                    token.checkpoint()
                path = prepared[task.id].path
                snapshot = manager.capture_snapshot(path)
                changed_files = tuple(snapshot.changed_files)
                violations = _ownership_violations(changed_files, task.files)
                error = getattr(result, "error", "") or ""
                verified, error = _annotate_verification(
                    error,
                    result.passed,
                    verification_kind,
                    allow_weak=allow_weak_verification,
                )
                passed = verified and bool(changed_files) and not violations
                if result.passed and not changed_files:
                    no_change = (
                        "verification passed, but the required subtask produced no changes"
                    )
                    error = f"{error}; {no_change}" if error else no_change
                elif violations:
                    violation = (
                        "worker changed files outside its ownership: "
                        + ", ".join(violations)
                    )
                    error = f"{error}; {violation}" if error else violation
                if run_context is not None:
                    run_context.task_status(
                        task.id,
                        task.description,
                        TaskStatus.SUCCEEDED if passed else TaskStatus.FAILED,
                        model=models_used[-1] if models_used else "",
                        owned_files=task.files,
                        worktree_path=path,
                        error=error,
                        metadata={"changed_files": list(changed_files)},
                    )
                return (
                    task,
                    path,
                    passed,
                    tuple(models_used),
                    changed_files,
                    violations,
                    error,
                )
            except RunCancelled:
                raise
            except Exception as exc:
                if run_context is not None:
                    run_context.task_status(
                        task.id,
                        task.description,
                        TaskStatus.FAILED,
                        owned_files=task.files,
                        worktree_path=prepared[task.id].path,
                        error=str(exc),
                    )
                return task, prepared[task.id].path, False, (), (), (), str(exc)

        # pool.map preserves task order, so the integration below stays deterministic.
        with ThreadPoolExecutor(max_workers=min(len(tasks), _MAX_PARALLEL)) as pool:
            ran = list(pool.map(_build, tasks))

        if token is not None:
            token.checkpoint()

        # 2. Merge the passing subtasks' diffs onto one integration worktree.
        integration = manager.prepare("_integration")
        if run_context is not None:
            run_context.set_worktree(integration.path)
        for task, path, sub_passed, models, changed_files, violations, error in ran:
            if token is not None:
                token.checkpoint()
            integrated = False
            if sub_passed:
                integrated = manager.apply_patch(integration.path, manager.diff_patch(path))
                if on_progress:
                    on_progress(
                        "integrate",
                        f"{task.id}: {'merged' if integrated else 'CONFLICT, skipped'}",
                    )
                if not integrated and not error:
                    error = "integration conflict"
            if run_context is not None:
                run_context.task_status(
                    task.id,
                    task.description,
                    TaskStatus.INTEGRATED if integrated else TaskStatus.FAILED,
                    model=models[-1] if models else "",
                    owned_files=task.files,
                    worktree_path=path,
                    error=error if not integrated else "",
                    metadata={
                        "changed_files": list(changed_files),
                        "integration_worktree": integration.path,
                    },
                )
            subs.append(
                FanoutSub(
                    id=task.id,
                    description=task.description,
                    passed=sub_passed,
                    integrated=integrated,
                    models_used=models,
                    required=task.required,
                    changed_files=changed_files,
                    ownership_violations=violations,
                    error=error if not integrated else "",
                )
            )

        # 3. Verify the merged whole -- passing means the tests are green AND at
        #    least one subtask actually landed.
        if on_progress:
            on_progress("verifying", f"running: {test_cmd}")
        if token is None:
            _output, returncode = _run_tests_in(test_cmd, integration.path, timeout)
        else:
            _output, returncode = _run_tests_in(
                test_cmd, integration.path, timeout, token,
            )
        merged_any = any(s.integrated for s in subs)

        snapshot = manager.capture_snapshot(integration.path)
        required_complete = all(
            not sub.required or (sub.passed and sub.integrated) for sub in subs
        )
        passed = (
            returncode == 0
            and required_complete
            and merged_any
            and bool(snapshot.changed_files)
        )
        if passed:
            outcome = RunOutcome.SUCCEEDED
        elif merged_any or any(sub.passed for sub in subs):
            outcome = RunOutcome.PARTIAL
        else:
            outcome = RunOutcome.FAILED
        errors = []
        if returncode != 0:
            errors.append("integrated verification failed")
        if not snapshot.changed_files:
            errors.append("fanout produced no repository changes")
        incomplete = [
            sub.id for sub in subs if sub.required and not (sub.passed and sub.integrated)
        ]
        if incomplete:
            errors.append("required subtasks incomplete: " + ", ".join(incomplete))
        # Only the reviewable integration worktree survives. Per-task and base
        # worktrees are scratch state and otherwise accumulate after every fanout.
        manager.cleanup(keep_provider="_integration")
        return FanoutResult(
            objective=objective,
            provider=provider_name,
            worktree_path=integration.path,
            subs=tuple(subs),
            passed=passed,
            outcome=outcome,
            verification_kind=verification_kind,
            diff_stat=snapshot.diff_stat,
            diff_excerpt=snapshot.diff_excerpt,
            changed_files=snapshot.changed_files,
            input_tokens=token_totals[0],
            output_tokens=token_totals[1],
            cost=cost_total[0],
            tokens_by_provider=_token_breakdown(),
            cost_by_provider=tuple(cost_by_provider.items()),
            error="; ".join(errors),
        )
    except RunCancelled as exc:
        snapshot = None
        worktree_path = integration.path if integration is not None else ""
        if integration is not None:
            try:
                snapshot = manager.capture_snapshot(integration.path)
            except Exception:
                pass
            manager.cleanup(keep_provider="_integration")
        else:
            manager.cleanup()
        return FanoutResult(
            objective=objective,
            provider=provider_name,
            worktree_path=worktree_path,
            subs=tuple(subs),
            passed=False,
            outcome=RunOutcome.CANCELLED,
            verification_kind=verification_kind,
            diff_stat=snapshot.diff_stat if snapshot else "",
            diff_excerpt=snapshot.diff_excerpt if snapshot else "",
            changed_files=snapshot.changed_files if snapshot else (),
            input_tokens=token_totals[0],
            output_tokens=token_totals[1],
            cost=cost_total[0],
            tokens_by_provider=_token_breakdown(),
            cost_by_provider=tuple(cost_by_provider.items()),
            error=str(exc),
        )
    except Exception as exc:
        worktree_path = integration.path if integration is not None else ""
        if integration is not None:
            manager.cleanup(keep_provider="_integration")
        else:
            manager.cleanup()
        return FanoutResult(
            objective=objective,
            provider=provider_name,
            worktree_path=worktree_path,
            subs=tuple(subs),
            passed=False,
            outcome=RunOutcome.FAILED,
            verification_kind=verification_kind,
            input_tokens=token_totals[0],
            output_tokens=token_totals[1],
            cost=cost_total[0],
            tokens_by_provider=_token_breakdown(),
            cost_by_provider=tuple(cost_by_provider.items()),
            error=str(exc),
        )
