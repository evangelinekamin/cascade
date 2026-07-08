"""run_fanout: the parallel verified orchestrator (build-mode north star).

A frontier "director" decomposes an objective into INDEPENDENT subtasks that each
own a distinct set of files; every subtask runs as a verified worker (with the
same bulk->frontier / cross-provider escalation as /solve) in its OWN isolated
worktree, and the passing subtasks' diffs are replayed onto one integration
worktree, which is then re-verified as a whole.

Where /pipeline is sequential -- one evolving worktree, each step building on the
last -- this is a fan-out: independent subtasks that don't see each other, then a
merge. Non-destructive: the caller's working tree is never touched.

MVP note: subtasks run sequentially here because run_verified_task swaps
``provider.config.model`` in place, which is not safe to share across threads.
True parallelism is a drop-in: give each subtask its own provider instance and
run them through a thread pool. The decompose / integrate / verify shape is the
load-bearing part, and it is parallel-ready.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Callable, List, Optional

from .solve import _resolve_escalation, _run_tests_in, _test_command, run_verified_task
from .worktree import WorktreeManager

ProgressCallback = Optional[Callable[[str, str], None]]

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
    {"id": "sub_1", "description": "short summary", "prompt": "the full self-contained instruction", "files": ["path/it/owns.py"]}
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


@dataclass(frozen=True)
class FanoutSub:
    """The outcome of one subtask and whether it merged into the integration."""

    id: str
    description: str
    passed: bool  # the subtask's own tests passed in its worktree
    integrated: bool  # its verified diff applied cleanly into the integration worktree
    models_used: tuple[str, ...] = ()


@dataclass(frozen=True)
class FanoutResult:
    """Outcome of a full parallel fan-out."""

    objective: str
    provider: str
    worktree_path: str  # the integration worktree
    subs: tuple[FanoutSub, ...]
    passed: bool  # the integrated whole passes the test suite
    diff_stat: str = ""
    diff_excerpt: str = ""
    changed_files: tuple[str, ...] = ()
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
            )
        )
    return (tasks or fallback)[:_MAX_SUBTASKS]


def plan_subtasks(
    app,
    objective: str,
    director,
    director_model: str,
    on_progress: ProgressCallback = None,
) -> List[FanoutTask]:
    """Ask the *director* (a frontier model) to split the objective into subtasks."""
    if director is None:
        return [FanoutTask(id="sub_1", description=objective, prompt=objective)]
    if on_progress:
        on_progress("planning", f"director ({director_model}) decomposing into subtasks")

    original_model = director.config.model
    director.config.model = director_model
    try:
        response = director.ask_single(
            f"Decompose this objective into independent parallel subtasks:\n\n{objective}",
            system=_DIRECTOR_SYSTEM,
        )
    except Exception:
        return [FanoutTask(id="sub_1", description=objective, prompt=objective)]
    finally:
        director.config.model = original_model

    return _parse_subtasks(objective, response)


def _clone_provider(provider):
    """A fresh provider instance with a COPIED config, safe to mutate in one thread.

    Subtasks build in parallel and run_verified_task swaps ``config.model`` in place,
    so they must not share a provider -- or even a config object. Best-effort: if the
    config is not a dataclass (e.g. a test double), fall back to the original.
    """
    if provider is None:
        return None
    try:
        return type(provider)(replace(provider.config))
    except Exception:
        return provider


def run_fanout(
    app,
    objective: str,
    provider_name: Optional[str] = None,
    *,
    max_iterations: int = 3,
    escalate: bool = True,
    escalate_after: int = 1,
    timeout: int = 300,
    on_progress: ProgressCallback = None,
) -> FanoutResult:
    """Decompose *objective* into independent subtasks, build each verified in its
    own worktree, merge the passing diffs, and re-verify the whole.
    """
    provider_name = provider_name or app.config.get_default_provider()
    provider = app.providers.get(provider_name)
    if provider is None:
        return FanoutResult(
            objective=objective,
            provider=provider_name,
            worktree_path="",
            subs=(),
            passed=False,
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
        director_model = escalation_model or escalation_provider.config.model
    else:
        director = provider
        director_model = frontier_model
    tasks = plan_subtasks(app, objective, director, director_model, on_progress=on_progress)
    test_cmd = _test_command(app)
    manager = WorktreeManager()

    try:
        # Preflight: a fan-out only makes sense on a green base. A red base forces
        # every subtask to also fix the same pre-existing failures, and those fixes
        # then collide on merge -- so check once up front and bail with a clear
        # message rather than producing a confusing pile of conflicts.
        base = manager.prepare("_base")
        if on_progress:
            on_progress("preflight", f"checking the base is green: {test_cmd}")
        _base_output, base_rc = _run_tests_in(test_cmd, base.path, timeout)
        if base_rc != 0:
            return FanoutResult(
                objective=objective,
                provider=provider_name,
                worktree_path="",
                subs=(),
                passed=False,
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
        prepared = {task.id: manager.prepare(task.id) for task in tasks}

        def _build(task):
            if on_progress:
                on_progress("subtask", f"{task.id}: {task.description}")
            try:
                result, models_used, _providers = run_verified_task(
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
                )
                return task, prepared[task.id].path, result, tuple(models_used)
            except Exception:
                return task, prepared[task.id].path, SimpleNamespace(passed=False), ()

        # pool.map preserves task order, so the integration below stays deterministic.
        with ThreadPoolExecutor(max_workers=min(len(tasks), _MAX_PARALLEL)) as pool:
            ran = list(pool.map(_build, tasks))

        # 2. Merge the passing subtasks' diffs onto one integration worktree.
        integration = manager.prepare("_integration")
        subs: List[FanoutSub] = []
        for task, path, result, models in ran:
            integrated = False
            if result.passed:
                integrated = manager.apply_patch(integration.path, manager.diff_patch(path))
                if on_progress:
                    on_progress(
                        "integrate",
                        f"{task.id}: {'merged' if integrated else 'CONFLICT, skipped'}",
                    )
            subs.append(
                FanoutSub(
                    id=task.id,
                    description=task.description,
                    passed=result.passed,
                    integrated=integrated,
                    models_used=models,
                )
            )

        # 3. Verify the merged whole -- passing means the tests are green AND at
        #    least one subtask actually landed.
        if on_progress:
            on_progress("verifying", f"running: {test_cmd}")
        _output, returncode = _run_tests_in(test_cmd, integration.path, timeout)
        merged_any = any(s.integrated for s in subs)
        passed = returncode == 0 and merged_any

        snapshot = manager.capture_snapshot(integration.path)
        return FanoutResult(
            objective=objective,
            provider=provider_name,
            worktree_path=integration.path,
            subs=tuple(subs),
            passed=passed,
            diff_stat=snapshot.diff_stat,
            diff_excerpt=snapshot.diff_excerpt,
            changed_files=snapshot.changed_files,
        )
    except Exception as exc:
        return FanoutResult(
            objective=objective,
            provider=provider_name,
            worktree_path="",
            subs=(),
            passed=False,
            error=str(exc),
        )
