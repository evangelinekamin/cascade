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
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from .solve import _test_command, run_verified_task
from .worktree import WorktreeManager

ProgressCallback = Optional[Callable[[str, str], None]]


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
    models_used: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of a full sequential verified pipeline."""

    objective: str
    provider: str
    worktree_path: str
    steps: tuple[PipelineStep, ...]
    passed: bool
    diff_stat: str = ""
    diff_excerpt: str = ""
    changed_files: tuple[str, ...] = ()
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
    for index, raw in enumerate(data.get("steps", []), start=1):
        if not isinstance(raw, dict):
            continue
        description = str(raw.get("description", "")).strip()
        prompt = str(raw.get("prompt", "")).strip() or description
        if not prompt:
            continue
        steps.append(
            PipelineTask(
                id=str(raw.get("id", f"step_{index}")),
                description=description or prompt,
                prompt=prompt,
            )
        )
    return steps or fallback


def plan_steps(
    app,
    objective: str,
    provider_name: str,
    on_progress: ProgressCallback = None,
) -> List[PipelineTask]:
    """Ask the director (on its frontier model) to decompose the objective."""
    provider = app.providers.get(provider_name)
    if provider is None:
        return [PipelineTask(id="step_1", description=objective, prompt=objective)]
    if on_progress:
        on_progress("planning", f"{provider_name} decomposing the objective")

    original_model = provider.config.model
    provider.config.model = app.config.get_model_for(provider_name, fast=False)
    try:
        response = provider.ask_single(
            f"Decompose this objective into ordered coding steps:\n\n{objective}",
            system=_PLANNER_SYSTEM,
        )
    except Exception:
        return [PipelineTask(id="step_1", description=objective, prompt=objective)]
    finally:
        provider.config.model = original_model

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
    on_progress: ProgressCallback = None,
) -> PipelineResult:
    """Decompose *objective* and build it step by step, each step test-verified.

    All steps run in one shared worktree, so each builds on the prior step's
    verified state. The pipeline passes when the final step leaves the test
    suite green. The caller's working tree is never touched.
    """
    provider_name = provider_name or app.config.get_default_provider()
    provider = app.providers.get(provider_name)
    if provider is None:
        return PipelineResult(
            objective=objective,
            provider=provider_name,
            worktree_path="",
            steps=(),
            passed=False,
            error=f"Provider '{provider_name}' not available",
        )

    tasks = plan_steps(app, objective, provider_name, on_progress=on_progress)
    frontier_model = app.config.get_model_for(provider_name, fast=False)
    bulk_model = (
        app.config.get_model_for(provider_name, fast=True) if escalate else frontier_model
    )
    test_cmd = _test_command(app)
    manager = WorktreeManager()

    try:
        path = manager.prepare(provider_name).path
        if on_progress:
            on_progress("workspace", path)

        completed: List[PipelineTask] = []
        step_results: List[PipelineStep] = []
        for task in tasks:
            if on_progress:
                on_progress("step", f"{task.id}: {task.description}")
            result, models_used = run_verified_task(
                provider,
                path,
                _step_prompt(objective, task, completed),
                test_cmd,
                bulk_model=bulk_model,
                frontier_model=frontier_model,
                max_iterations=max_iterations,
                escalate=escalate,
                escalate_after=escalate_after,
                timeout=timeout,
                on_progress=on_progress,
            )
            step_results.append(
                PipelineStep(
                    id=task.id,
                    description=task.description,
                    passed=result.passed,
                    iterations=result.iterations,
                    models_used=tuple(models_used),
                )
            )
            completed.append(task)

        snapshot = manager.capture_snapshot(path)
        final_passed = bool(step_results) and step_results[-1].passed
        return PipelineResult(
            objective=objective,
            provider=provider_name,
            worktree_path=path,
            steps=tuple(step_results),
            passed=final_passed,
            diff_stat=snapshot.diff_stat,
            diff_excerpt=snapshot.diff_excerpt,
            changed_files=snapshot.changed_files,
        )
    except Exception as exc:
        return PipelineResult(
            objective=objective,
            provider=provider_name,
            worktree_path="",
            steps=(),
            passed=False,
            error=str(exc),
        )
