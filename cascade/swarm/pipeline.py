"""run_pipeline: the sequential verified orchestrator.

A frontier "director" decomposes a large objective into an ordered list of
coding steps; each step runs as a verified worker (with bulk->frontier
escalation) in ONE shared git worktree, so every step builds on the previous
step's test-verified state. There is no cross-worker merge -- a single evolving
worktree -- which keeps it correct and simple. Non-destructive: the caller's
working tree is never touched.
"""

from __future__ import annotations

import json
import inspect
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, List, Optional

from .lifecycle import CancellationToken, RunCancelled, RunContext, TaskStatus
from ..providers.usage import Usage
from .outcome import RunOutcome
from .solve import (
    _annotate_verification,
    _resolve_escalation,
    _test_command,
    classify_verification,
    run_verified_task,
)
from .worktree import WorktreeManager

ProgressCallback = Optional[Callable[[str, str], None]]
TokensCallback = Optional[Callable[[int, int], None]]


_PLANNER_SYSTEM = """\
You are a software director. Decompose the objective into an ORDERED list of
small, concrete coding steps that build on each other. Each step is one focused
change that can be made in the workspace and checked by running the test suite.
Earlier steps establish foundations later steps depend on; where it makes sense,
an early step writes the tests that later steps must satisfy.

Respond with JSON only:
{
  "steps": [
    {"id": "step_1", "description": "short summary", "prompt": "the full instruction for this step"}
  ]
}

Keep it to 2-6 steps. Order matters.
"""

_STEP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class PipelineTask:
    """One planned step in the pipeline."""

    id: str
    description: str
    prompt: str


@dataclass(frozen=True)
class PipelineStep:
    """The verified outcome of running one PipelineTask."""

    id: str
    description: str
    passed: bool
    iterations: int
    changed: bool = False
    models_used: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of a full sequential verified pipeline."""

    objective: str
    provider: str
    worktree_path: str
    steps: tuple[PipelineStep, ...]
    passed: bool
    outcome: RunOutcome = RunOutcome.FAILED
    verification_kind: str = ""
    diff_stat: str = ""
    diff_excerpt: str = ""
    changed_files: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    # Per-provider breakdown, same shape as SolveResult's: an escalated step's
    # tokens and cost belong to the model that incurred them, not to the base
    # provider. (provider, input_tokens, output_tokens) and (provider, cost).
    tokens_by_provider: tuple[tuple[str, int, int], ...] = ()
    cost_by_provider: tuple[tuple[str, float], ...] = ()
    error: str = ""


def _parse_steps(objective: str, response: str) -> List[PipelineTask]:
    """Parse the director's JSON into ordered PipelineTasks, with a fallback."""
    fallback = [PipelineTask(id="step_1", description=objective, prompt=objective)]
    match = re.search(r"\{[\s\S]*\}", response or "")
    if not match:
        return fallback
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return fallback

    steps: List[PipelineTask] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(data.get("steps", []), start=1):
        if not isinstance(raw, dict):
            continue
        description = str(raw.get("description", "")).strip()
        prompt = str(raw.get("prompt", "")).strip() or description
        if not prompt:
            continue
        task_id = str(raw.get("id", f"step_{index}")).strip()
        if not _STEP_ID_RE.fullmatch(task_id) or task_id in seen_ids:
            task_id = f"step_{index}"
            suffix = 2
            while task_id in seen_ids:
                task_id = f"step_{index}_{suffix}"
                suffix += 1
        seen_ids.add(task_id)
        steps.append(
            PipelineTask(
                id=task_id,
                description=description or prompt,
                prompt=prompt,
            )
        )
    return steps or fallback


def _director_for(app, provider_name: str) -> "tuple[Optional[object], str, Optional[str]]":
    """The planner: the configured escalation target, else the primary provider.

    Planning is the one frontier-model call in a pipeline, so run_pipeline needs
    the same answer plan_steps acts on to bill it to the right provider.
    """
    target = app.config.get_escalation_target(provider_name)
    if not isinstance(target, (str, tuple, list)):
        target = None
    director, director_name, director_model = _resolve_escalation(app, target)
    if director is not None:
        return director, director_name, director_model
    return (
        app.providers.get(provider_name),
        provider_name,
        app.config.get_model_for(provider_name, fast=False),
    )


def plan_steps(
    app,
    objective: str,
    provider_name: str,
    on_progress: ProgressCallback = None,
    on_tokens: TokensCallback = None,
    on_cost: Optional[Callable[[float], None]] = None,
    cancel_token: Optional[CancellationToken] = None,
    context: str = "",
) -> List[PipelineTask]:
    """Ask the director (on its frontier model) to decompose the objective.

    ``context`` is the same bounded conversation digest the workers receive (see
    :func:`cascade.conversation.build_lane_context`); prepended here so the
    director can resolve a referential objective ("fix the errors codex found")
    before splitting it into steps, not just each worker after the fact.
    """
    if app.providers.get(provider_name) is None:
        return [PipelineTask(id="step_1", description=objective, prompt=objective)]
    director, director_name, director_model = _director_for(app, provider_name)
    if on_progress:
        on_progress(
            "planning",
            f"{director_name} ({director_model}) decomposing the objective",
        )

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
        planner_prompt = f"Decompose this objective into ordered coding steps:\n\n{objective}"
        if context:
            planner_prompt = f"{context}\n\n{planner_prompt}"
        with scope:
            response = director.ask_single(
                planner_prompt,
                system=_PLANNER_SYSTEM,
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
        return [PipelineTask(id="step_1", description=objective, prompt=objective)]
    finally:
        director.config.model = original_model

    return _parse_steps(objective, response)


def _step_prompt(objective: str, task: PipelineTask, completed: List[PipelineTask]) -> str:
    """Build the prompt for one step, with context from completed steps."""
    parts = [f"Overall objective:\n{objective}"]
    if completed:
        done = "\n".join(f"- {t.id}: {t.description}" for t in completed)
        parts.append("Already completed (their changes are in the workspace):\n" + done)
    parts.append(f"Current step ({task.id}): {task.description}")
    parts.append(task.prompt)
    return "\n\n".join(parts)


def run_pipeline(
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
    context: str = "",
) -> PipelineResult:
    """Decompose *objective* and build it step by step, each step test-verified.

    All steps run in one shared worktree, so each builds on the prior step's
    verified state. The pipeline passes when the final step leaves the test
    suite green. The caller's working tree is never touched.
    """
    provider_name = provider_name or app.config.get_default_provider()
    token = cancel_token or (run_context.token if run_context is not None else None)
    if token is not None:
        token.checkpoint()
    if run_context is not None:
        run_context.start(workflow="pipeline", provider=provider_name)
    provider = app.providers.get(provider_name)
    if provider is None:
        return PipelineResult(
            objective=objective,
            provider=provider_name,
            worktree_path="",
            steps=(),
            passed=False,
            outcome=RunOutcome.BLOCKED,
            error=f"Provider '{provider_name}' not available",
        )

    token_totals = [0, 0]
    cost_total = [0.0]
    tokens_by_provider: dict[str, list[int]] = {}
    cost_by_provider: dict[str, float] = {}

    def _accumulate_tokens(input_tokens: int, output_tokens: int) -> None:
        token_totals[0] += input_tokens
        token_totals[1] += output_tokens
        if run_context is not None:
            run_context.add_tokens(input_tokens, output_tokens)
        if on_tokens is not None:
            on_tokens(input_tokens, output_tokens)

    def _accumulate_cost(cost: float) -> None:
        cost_total[0] += cost
        if run_context is not None:
            run_context.add_cost(cost)

    def _credit_tokens(label: str, input_tokens: int, output_tokens: int) -> None:
        bucket = tokens_by_provider.setdefault(label, [0, 0])
        bucket[0] += input_tokens
        bucket[1] += output_tokens

    def _credit_cost(label: str, cost: float) -> None:
        cost_by_provider[label] = cost_by_provider.get(label, 0.0) + cost

    def _token_breakdown() -> "tuple[tuple[str, int, int], ...]":
        return tuple(
            (label, tokens[0], tokens[1]) for label, tokens in tokens_by_provider.items()
        )

    def _credit_step(attribution: list) -> None:
        """Fold one step's per-provider breakdown into the run-wide one."""
        for label, step_in, step_out in (attribution[0] if attribution else ()):
            _credit_tokens(label, step_in, step_out)
        for label, step_cost in (attribution[1] if len(attribution) > 1 else ()):
            _credit_cost(label, step_cost)

    _director, director_name, _director_model = _director_for(app, provider_name)

    def _plan_tokens(input_tokens: int, output_tokens: int) -> None:
        """Planning runs on the director's provider, not on a step worker's."""
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
    plan_params = inspect.signature(plan_steps).parameters
    if "on_cost" in plan_params:
        plan_kwargs["on_cost"] = _plan_cost
    if "context" in plan_params:
        plan_kwargs["context"] = context
    tasks = plan_steps(app, objective, provider_name, **plan_kwargs)
    if run_context is not None:
        for index, task in enumerate(tasks):
            depends_on = (tasks[index - 1].id,) if index else ()
            run_context.declare_task(
                task.id, task.description, depends_on=depends_on,
            )
    frontier_model = app.config.get_model_for(provider_name, fast=False)
    bulk_model = app.config.get_bulk_model(provider_name) if escalate else frontier_model
    target = app.config.get_escalation_target(provider_name) if escalate else None
    if not isinstance(target, (str, tuple, list)):
        target = None
    escalation_provider, escalation_name, escalation_model = _resolve_escalation(
        app, target
    )
    test_cmd = _test_command(app)
    verification_kind = classify_verification(test_cmd)
    manager = WorktreeManager()
    path = ""
    step_results: List[PipelineStep] = []

    try:
        if token is not None:
            token.checkpoint()
        path = manager.prepare(provider_name).path
        if run_context is not None:
            run_context.set_worktree(path)
        if on_progress:
            on_progress("workspace", path)

        completed: List[PipelineTask] = []
        for task_index, task in enumerate(tasks):
            if token is not None:
                token.checkpoint()
            depends_on = (tasks[task_index - 1].id,) if task_index else ()
            if run_context is not None:
                run_context.task_status(
                    task.id,
                    task.description,
                    TaskStatus.RUNNING,
                    depends_on=depends_on,
                    worktree_path=path,
                )
            if on_progress:
                on_progress("step", f"{task.id}: {task.description}")
            before_patch = manager.diff_patch(path)
            result, models_used, _providers_used, *attribution = run_verified_task(
                provider,
                path,
                _step_prompt(objective, task, completed),
                test_cmd,
                bulk_model=bulk_model,
                frontier_model=frontier_model,
                max_iterations=max_iterations,
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
                cancel_token=token,
                context=context,
            )
            _credit_step(attribution)
            if token is not None:
                token.checkpoint()
            step_error = getattr(result, "error", "") or ""
            verified, step_error = _annotate_verification(
                step_error,
                result.passed,
                verification_kind,
                allow_weak=allow_weak_verification,
            )
            changed = manager.diff_patch(path) != before_patch
            step_passed = verified and changed
            if result.passed and not changed:
                no_change = "verification passed, but this required step produced no changes"
                step_error = f"{step_error}; {no_change}" if step_error else no_change
            step_results.append(
                PipelineStep(
                    id=task.id,
                    description=task.description,
                    passed=step_passed,
                    iterations=result.iterations,
                    changed=changed,
                    models_used=tuple(models_used),
                    error=step_error,
                )
            )
            if run_context is not None:
                run_context.task_status(
                    task.id,
                    task.description,
                    TaskStatus.SUCCEEDED if step_passed else TaskStatus.FAILED,
                    model=models_used[-1] if models_used else "",
                    depends_on=depends_on,
                    worktree_path=path,
                    error=step_error,
                    metadata={"iterations": result.iterations, "changed": changed},
                )
            if not step_passed:
                if on_progress:
                    on_progress("stopped", f"{task.id} failed; dependent steps were not run")
                if run_context is not None:
                    for skipped_index, skipped in enumerate(tasks[task_index + 1:], start=task_index + 1):
                        run_context.task_status(
                            skipped.id,
                            skipped.description,
                            TaskStatus.SKIPPED,
                            depends_on=(tasks[skipped_index - 1].id,),
                            worktree_path=path,
                            error=f"dependency {task.id} did not succeed",
                        )
                break
            completed.append(task)

        snapshot = manager.capture_snapshot(path)
        final_passed = (
            len(step_results) == len(tasks)
            and bool(step_results)
            and all(step.passed for step in step_results)
            and bool(snapshot.changed_files)
        )
        error = ""
        if not final_passed:
            failed = next((step for step in step_results if not step.passed), None)
            if failed is not None:
                error = f"step {failed.id} failed: {failed.error or 'verification failed'}"
            elif not snapshot.changed_files:
                error = "pipeline produced no repository changes"
        if final_passed:
            outcome = RunOutcome.SUCCEEDED
        elif any(step.passed for step in step_results) and snapshot.changed_files:
            outcome = RunOutcome.PARTIAL
        else:
            outcome = RunOutcome.FAILED
        return PipelineResult(
            objective=objective,
            provider=provider_name,
            worktree_path=path,
            steps=tuple(step_results),
            passed=final_passed,
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
            error=error,
        )
    except RunCancelled as exc:
        snapshot = None
        if path:
            try:
                snapshot = manager.capture_snapshot(path)
            except Exception:
                pass
        return PipelineResult(
            objective=objective,
            provider=provider_name,
            worktree_path=path,
            steps=tuple(step_results),
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
        return PipelineResult(
            objective=objective,
            provider=provider_name,
            worktree_path=path,
            steps=tuple(step_results),
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
