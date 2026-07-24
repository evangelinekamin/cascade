"""Model-selected orchestration for normal Cascade prompts.

The classifier is a cheap control-plane call; it never edits files.  Execution is
then delegated to one of the capability-constrained paths: normal chat, read-only
reconnaissance, a focused verified solve, a sequential pipeline, or validated
parallel fanout. Slash commands remain explicit debug/override controls.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional

from .fanout import run_fanout
from .lifecycle import CancellationToken, RunCancelled, RunContext, TaskStatus
from ..providers.usage import Usage
from .outcome import RunOutcome
from .pipeline import run_pipeline
from .solve import run_solve
from .workspace import WorkspaceTools


class WorkflowKind(str, Enum):
    CHAT = "chat"
    RECON = "recon"
    SOLVE = "solve"
    PIPELINE = "pipeline"
    FANOUT = "fanout"


@dataclass(frozen=True)
class RouteDecision:
    workflow: WorkflowKind
    reason: str
    confidence: float
    worker_tier: str = "bulk"
    router_provider: str = "heuristic"
    router_model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass(frozen=True)
class AutoResult:
    decision: RouteDecision
    outcome: RunOutcome
    text: str
    execution_provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


ProgressCallback = Optional[Callable[[str, str], None]]

_RECON_READONLY_SYSTEM = (
    "Inspect the repository to answer the request. You have only read-only "
    "tools. Cite file paths and concrete evidence; do not claim to have changed "
    "anything."
)

# Test mode: the goal is to CONFIRM the project works, so running its checks is
# the whole point. The permission gate still stops anything destructive.
_RECON_VERIFY_SYSTEM = (
    "Verify whether this project actually works, and report concrete pass/fail "
    "evidence. You MAY run its tests, type-checks, linters, and build (e.g. "
    "'npm test', 'npm run build', 'pytest', 'tsc --noEmit') -- running them is "
    "how you confirm it works, not a guess from reading. Prefer the project's "
    "own configured commands. Do NOT edit source files; build artifacts and "
    "temp files are expected. Cite file paths and the actual command output."
)


_ROUTER_SYSTEM = """\
You route software-assistant requests. Choose the smallest workflow that can
complete the request safely:

- chat: conversation, ideation, or a question that needs no repository access.
- recon: read/search/trace/review the repository without changing any file.
- solve: one focused code change that can be verified in one worktree.
- pipeline: multiple dependent code changes that must build on one another.
- fanout: multiple genuinely independent code changes with disjoint file ownership.

Prefer solve over pipeline when uncertain. Choose fanout only when independence is
explicit and credible; parallelism is not a goal by itself. Never infer a code edit
from a request that only asks for an explanation or review.

Also choose worker_tier:
- fast: a tiny, localized, low-blast-radius solve that tests can verify; this may use
  a very fast but less reliable model and will escalate if it fails.
- bulk: normal verified implementation work.
- frontier: unusually difficult or ambiguous implementation work.
Use bulk unless fast or frontier is clearly justified. The tier is ignored for chat,
recon, pipeline, and fanout.
"""

_ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "workflow": {
            "type": "string",
            "enum": [kind.value for kind in WorkflowKind],
            "description": "The smallest safe workflow for the request",
        },
        "reason": {
            "type": "string",
            "description": "One short sentence explaining the routing decision",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "worker_tier": {
            "type": "string",
            "enum": ["fast", "bulk", "frontier"],
            "description": "Initial worker strength for a focused solve",
        },
    },
    "required": ["workflow", "reason", "confidence", "worker_tier"],
    "additionalProperties": False,
}

_READ_RE = re.compile(
    r"\b(read|inspect|review|explain|trace|find|search|locate|understand|"
    r"investigate|analy[sz]e|audit|what|where|why|how)\b",
    re.IGNORECASE,
)
_EDIT_RE = re.compile(
    r"\b(add|build|change|create|delete|edit|fix|implement|migrate|modify|"
    r"refactor|remove|rename|replace|update|write|wire)\b",
    re.IGNORECASE,
)
_DEPENDENT_RE = re.compile(
    r"\b(across|end[- ]to[- ]end|multi[- ]step|migration|refactor|throughout|"
    r"all (?:modules|packages|layers)|backend and frontend|schema and)\b",
    re.IGNORECASE,
)
_PARALLEL_RE = re.compile(
    r"\b(in parallel|independent(?:ly)?|disjoint|unrelated|separate modules|"
    r"separate packages)\b",
    re.IGNORECASE,
)
_FAST_EDIT_RE = re.compile(
    r"\b(typo|one[- ]line|tiny|small localized|formatting only)\b",
    re.IGNORECASE,
)


def _is_git_worktree() -> bool:
    """Whether the launch directory sits inside a git work tree.

    Every non-chat workflow builds in a git worktree, so orchestration is only
    meaningful inside a repository. Outside one, prompts must fall back to
    ordinary chat rather than die on a raw ``fatal: not a git repository``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=2,
        )
    except Exception:
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def should_auto_orchestrate(app, mode: str) -> bool:
    """Return whether ordinary prompts in *mode* should be model-routed."""
    try:
        config = app.config.get_orchestration_config()
    except Exception:
        return False
    if not (bool(config.get("enabled")) and mode in config.get("modes", ())):
        return False
    # All routed workflows build in a git worktree; outside a repository there is
    # nothing to orchestrate, so let the prompt fall through to ordinary chat.
    return _is_git_worktree()


def _lane_provider(app, provider_name: str, model: str, preferences: dict):
    """Clone a configured provider for a control/recon lane without mutating chat."""
    original = app.providers.get(provider_name)
    if original is None:
        return None
    try:
        config = replace(
            original.config,
            model=model,
            provider_preferences=dict(preferences),
        )
        clone = type(original)(config)
        # Clones must keep the hook + permission gates they were wired with.
        clone.hook_runner = getattr(original, "hook_runner", None)
        clone.permission_engine = getattr(original, "permission_engine", None)
        return clone
    except Exception:
        return None


def _parse_route_payload(payload: object) -> tuple[WorkflowKind, str, float, str]:
    if isinstance(payload, str):
        match = re.search(r"\{[\s\S]*\}", payload)
        if not match:
            raise ValueError("router returned no JSON object")
        payload = json.loads(match.group())
    if not isinstance(payload, dict):
        raise ValueError("router response was not an object")
    workflow = WorkflowKind(str(payload.get("workflow", "")).strip().lower())
    reason = str(payload.get("reason", "")).strip() or "selected by the workflow router"
    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    worker_tier = str(payload.get("worker_tier", "bulk")).strip().lower()
    if worker_tier not in ("fast", "bulk", "frontier"):
        worker_tier = "bulk"
    return workflow, reason, max(0.0, min(confidence, 1.0)), worker_tier


def _heuristic_route(prompt: str, mode: str, reason_prefix: str = "") -> RouteDecision:
    """Conservative fallback used when the configured routing model is unavailable."""
    reads = bool(_READ_RE.search(prompt))
    edits = bool(_EDIT_RE.search(prompt))
    if reads and not edits:
        workflow = WorkflowKind.RECON
        reason = "the request asks for repository inspection without an edit"
    elif edits and _PARALLEL_RE.search(prompt):
        workflow = WorkflowKind.FANOUT
        reason = "the request explicitly describes independent work"
    elif edits and _DEPENDENT_RE.search(prompt):
        workflow = WorkflowKind.PIPELINE
        reason = "the request describes multiple dependent changes"
    elif edits:
        workflow = WorkflowKind.SOLVE
        reason = "the request is a focused code change"
    else:
        workflow = WorkflowKind.CHAT
        reason = "the request does not clearly require an automated repository workflow"
    if reason_prefix:
        reason = f"{reason_prefix}; {reason}"
    tier = "fast" if workflow == WorkflowKind.SOLVE and _FAST_EDIT_RE.search(prompt) else "bulk"
    return RouteDecision(
        workflow=workflow,
        reason=reason,
        confidence=0.55,
        worker_tier=tier,
    )


def select_workflow(
    app,
    prompt: str,
    mode: str,
    cancel_token: Optional[CancellationToken] = None,
) -> RouteDecision:
    """Use the configured cheap model to select a workflow, failing conservatively."""
    if cancel_token is not None:
        cancel_token.checkpoint()
    config = app.config.get_orchestration_config()
    provider_name = config["router_provider"]
    model = config["router_model"]
    router = _lane_provider(
        app,
        provider_name,
        model,
        config["provider_preferences"],
    )
    if router is None:
        return _heuristic_route(prompt, mode, "routing model unavailable")

    try:
        request = f"Current Cascade mode: {mode}\n\nUser request:\n{prompt}"
        cancellation_scope = getattr(router, "cancellation_scope", None)
        scope = (
            cancellation_scope(cancel_token)
            if callable(cancellation_scope) and cancel_token is not None
            else nullcontext()
        )
        client = getattr(router, "client", None)
        close_callback = getattr(client, "close", None)
        callback_scope = (
            router.cancellation_callback(close_callback)
            if cancel_token is not None and callable(close_callback)
            else nullcontext()
        )
        with scope, callback_scope:
            ask_structured = getattr(router, "ask_structured", None)
            if callable(ask_structured):
                payload = ask_structured(
                    request,
                    _ROUTE_SCHEMA,
                    system=_ROUTER_SYSTEM,
                    schema_name="cascade_workflow_route",
                )
            else:
                payload = router.ask_single(
                    request
                    + "\n\nReturn JSON only with workflow, reason, confidence, and worker_tier.",
                    system=_ROUTER_SYSTEM,
                )
        if cancel_token is not None:
            cancel_token.checkpoint()
        workflow, reason, confidence, worker_tier = _parse_route_payload(payload)
        usage = router.last_usage or Usage()
        cost = usage.cost or 0.0
        return RouteDecision(
            workflow=workflow,
            reason=reason,
            confidence=confidence,
            worker_tier=worker_tier,
            router_provider=provider_name,
            router_model=model,
            input_tokens=usage.prompt_total,
            output_tokens=usage.output,
            cost=cost,
        )
    except RunCancelled:
        raise
    except Exception:
        if cancel_token is not None:
            cancel_token.checkpoint()
        return _heuristic_route(prompt, mode, "routing model failed")


def _route_header(decision: RouteDecision, actual: WorkflowKind | None = None) -> str:
    route = decision.workflow.value
    if actual is not None and actual != decision.workflow:
        route = f"{route} -> {actual.value}"
    model = f" via {decision.router_model}" if decision.router_model else ""
    tier = f" ({decision.worker_tier} worker)" if decision.workflow == WorkflowKind.SOLVE else ""
    return f"Auto route: {route}{tier}{model}\nReason: {decision.reason}"


def _format_solve(decision: RouteDecision, result) -> str:
    lines = [
        _route_header(decision),
        f"Solve {result.outcome.value.upper()} after {result.iterations} iteration(s) on {result.provider}",
    ]
    if result.models_used:
        lines.append("Models: " + " -> ".join(dict.fromkeys(result.models_used)))
    cost = float(getattr(result, "cost", 0.0) or 0.0)
    if cost:
        lines.append(f"OpenRouter cost: {cost:.6f} credits")
    if result.error:
        lines.append(f"Error: {result.error}")
    if result.diff_stat:
        lines.append(result.diff_stat.strip())
    elif result.changed_files:
        lines.append("Files: " + ", ".join(result.changed_files[:8]))
    if result.worktree_path:
        lines.append(f"Review + apply: git -C {result.worktree_path} diff")
    return "\n".join(lines)


def _format_pipeline(decision: RouteDecision, result, rerouted: bool = False) -> str:
    lines = [
        _route_header(decision, WorkflowKind.PIPELINE if rerouted else None),
        f"Pipeline {result.outcome.value.upper()}: "
        f"{sum(1 for step in result.steps if step.passed)}/{len(result.steps)} required steps verified",
    ]
    if result.error:
        lines.append(f"Error: {result.error}")
    cost = float(getattr(result, "cost", 0.0) or 0.0)
    if cost:
        lines.append(f"OpenRouter cost: {cost:.6f} credits")
    for step in result.steps:
        lines.append(
            f"  [{step.id}] {'OK' if step.passed else 'FAIL'} "
            f"({step.iterations} iter): {step.description}"
        )
        if step.error:
            lines.append(f"    {step.error}")
    if result.diff_stat:
        lines.append(result.diff_stat.strip())
    if result.worktree_path:
        lines.append(f"Review + apply: git -C {result.worktree_path} diff")
    return "\n".join(lines)


def _format_fanout(decision: RouteDecision, result) -> str:
    lines = [
        _route_header(decision),
        f"Fanout {result.outcome.value.upper()}: "
        f"{sum(1 for sub in result.subs if sub.integrated)}/{len(result.subs)} subtasks merged",
    ]
    if result.error:
        lines.append(f"Error: {result.error}")
    cost = float(getattr(result, "cost", 0.0) or 0.0)
    if cost:
        lines.append(f"OpenRouter cost: {cost:.6f} credits")
    for sub in result.subs:
        state = "MERGED" if sub.integrated else ("CONFLICT" if sub.passed else "FAIL")
        lines.append(f"  [{sub.id}] {state}: {sub.description}")
        if sub.error:
            lines.append(f"    {sub.error}")
    if result.diff_stat:
        lines.append(result.diff_stat.strip())
    if result.worktree_path:
        lines.append(f"Review + apply: git -C {result.worktree_path} diff")
    return "\n".join(lines)


def execute_auto(
    app,
    prompt: str,
    active_provider: str,
    decision: RouteDecision,
    *,
    mode: str = "",
    on_progress: ProgressCallback = None,
    on_tool_event=None,
    cancel_token: Optional[CancellationToken] = None,
    run_context: Optional[RunContext] = None,
) -> AutoResult:
    """Execute a non-chat route selected by :func:`select_workflow`."""
    token = cancel_token or (run_context.token if run_context is not None else None)

    def _checkpoint() -> None:
        if token is not None:
            token.checkpoint()

    _checkpoint()
    route_in = decision.input_tokens
    route_out = decision.output_tokens
    route_cost = decision.cost

    if decision.workflow == WorkflowKind.RECON:
        if run_context is not None:
            run_context.start(workflow=WorkflowKind.RECON.value)
            run_context.declare_task("recon", prompt)
            run_context.task_status("recon", prompt, TaskStatus.RUNNING)
        config = app.config.get_orchestration_config()
        provider_name = config["recon_provider"]
        provider = _lane_provider(
            app,
            provider_name,
            config["recon_model"],
            config["provider_preferences"],
        )
        disposable_provider = provider is not None
        if provider is None:
            # A direct-API active provider can still be constrained to read-only
            # tools. CLI proxies are excluded because their native tool set is not
            # controlled by WorkspaceTools.
            candidate = app.providers.get(active_provider)
            if candidate is not None and not (
                getattr(candidate, "_use_cli_proxy", False)
                or getattr(candidate, "_use_oauth_cli", False)
            ):
                provider = candidate
                provider_name = active_provider
                disposable_provider = False
        if provider is None:
            text = _route_header(decision) + "\nRecon BLOCKED: no safe read-only provider is available"
            if run_context is not None:
                run_context.task_status(
                    "recon", prompt, TaskStatus.BLOCKED,
                    error="no safe read-only provider is available",
                )
            return AutoResult(
                decision, RunOutcome.BLOCKED, text, provider_name,
                route_in, route_out, route_cost,
            )
        if on_progress:
            on_progress("recon", f"{provider_name} reading the repository")
        try:
            _checkpoint()
            cancellation_scope = getattr(provider, "cancellation_scope", None)
            scope = cancellation_scope(token) if callable(cancellation_scope) else nullcontext()
            client = getattr(provider, "client", None)
            close_callback = getattr(client, "close", None)
            callback_scope = (
                provider.cancellation_callback(close_callback)
                if disposable_provider and token is not None and callable(close_callback)
                else nullcontext()
            )
            # Test mode is about verification, which means actually RUNNING the
            # project's checks -- a read-only pass can only guess. So recon in
            # test mode gets run_command (permission-gated: transparent
            # test/build commands auto-approve, dangerous ones are denied) and a
            # prompt that permits running checks but not editing source.
            ws = WorkspaceTools(os.getcwd(), cancel_token=token)
            verify_mode = mode == "test"
            recon_tools = ws.build_verify() if verify_mode else ws.build_read_only()
            recon_system = _RECON_VERIFY_SYSTEM if verify_mode else _RECON_READONLY_SYSTEM
            if verify_mode:
                # A CLI-proxy provider (codex) runs its own sandbox, not our
                # tools, so read-only-tools alone won't let it execute; this
                # flag forces its writable sandbox. Harmless on direct-API
                # providers, which don't have it.
                setattr(provider, "_force_repo_write", True)
            with scope, callback_scope:
                response, _log = provider.ask_with_tools(
                    [{"role": "user", "content": prompt}],
                    recon_tools,
                    system=recon_system,
                    max_rounds=config["recon_max_rounds"],
                    on_tool_event=on_tool_event,
                )
            _checkpoint()
            usage = provider.last_usage or Usage()
            execution_cost = usage.cost or 0.0
            if run_context is not None:
                run_context.add_cost(execution_cost)
            outcome = RunOutcome.SUCCEEDED if response.strip() else RunOutcome.FAILED
            body = response.strip() or "Recon produced no response."
            text = _route_header(decision) + f"\nRecon {outcome.value.upper()} on {provider_name}\n\n{body}"
            if run_context is not None:
                run_context.task_status(
                    "recon",
                    prompt,
                    TaskStatus.SUCCEEDED if outcome == RunOutcome.SUCCEEDED else TaskStatus.FAILED,
                    model=provider.config.model,
                    error="" if outcome == RunOutcome.SUCCEEDED else "recon produced no response",
                )
            return AutoResult(
                decision,
                outcome,
                text,
                provider_name,
                route_in + usage.prompt_total,
                route_out + usage.output,
                route_cost + execution_cost,
            )
        except Exception as exc:
            if token is not None:
                token.checkpoint()
            text = _route_header(decision) + f"\nRecon FAILED on {provider_name}: {exc}"
            if run_context is not None:
                run_context.task_status(
                    "recon", prompt, TaskStatus.FAILED, error=str(exc),
                )
            return AutoResult(
                decision, RunOutcome.FAILED, text, provider_name,
                route_in, route_out, route_cost,
            )

    if decision.workflow == WorkflowKind.SOLVE:
        solve_provider = active_provider
        escalation_target = app.config.get_escalation_target(active_provider)
        bulk_model_override = None
        provider_preferences_override = None
        escalate = True
        if decision.worker_tier == "fast":
            config = app.config.get_orchestration_config()
            configured_fast_provider = config["fast_provider"]
            if configured_fast_provider in app.providers:
                solve_provider = configured_fast_provider
                bulk_model_override = config["fast_model"]
                provider_preferences_override = config["fast_provider_preferences"]
                if solve_provider != active_provider and active_provider in app.providers:
                    escalation_target = (
                        active_provider,
                        app.config.get_model_for(active_provider, fast=False),
                    )
                else:
                    escalation_target = app.config.get_escalation_target(solve_provider)
            else:
                bulk_model_override = app.config.get_model_for(active_provider, fast=True)
        elif decision.worker_tier == "frontier":
            target = escalation_target
            if isinstance(target, (tuple, list)) and target:
                target_name = str(target[0])
                target_model = target[1] if len(target) > 1 else None
            elif isinstance(target, str):
                target_name, target_model = target, None
            else:
                target_name, target_model = "", None
            if target_name in app.providers:
                solve_provider = target_name
                bulk_model_override = target_model or app.config.get_model_for(
                    target_name, fast=False
                )
                escalation_target = None
            escalate = False
        result = run_solve(
            app,
            prompt,
            provider_name=solve_provider,
            escalate=escalate,
            escalate_to=escalation_target,
            bulk_model_override=bulk_model_override,
            provider_preferences_override=provider_preferences_override,
            on_progress=on_progress,
            on_tool_event=on_tool_event,
            cancel_token=token,
            run_context=run_context,
        )
        result_cost = float(getattr(result, "cost", 0.0) or 0.0)
        return AutoResult(
            decision,
            result.outcome,
            _format_solve(decision, result),
            result.provider,
            route_in + result.input_tokens,
            route_out + result.output_tokens,
            route_cost + result_cost,
        )

    if decision.workflow == WorkflowKind.PIPELINE:
        result = run_pipeline(
            app,
            prompt,
            provider_name=active_provider,
            on_progress=on_progress,
            cancel_token=token,
            run_context=run_context,
        )
        result_cost = float(getattr(result, "cost", 0.0) or 0.0)
        return AutoResult(
            decision,
            result.outcome,
            _format_pipeline(decision, result),
            result.provider,
            route_in + result.input_tokens,
            route_out + result.output_tokens,
            route_cost + result_cost,
        )

    if decision.workflow == WorkflowKind.FANOUT:
        result = run_fanout(
            app,
            prompt,
            provider_name=active_provider,
            on_progress=on_progress,
            cancel_token=token,
            run_context=run_context,
        )
        # Unsafe/ambiguous parallel plans fail closed in fanout. Sequential work is
        # the safe automatic fallback and preserves progress toward the objective.
        if result.outcome == RunOutcome.BLOCKED and result.error.startswith("invalid parallel plan"):
            if on_progress:
                on_progress("rerouting", "parallel plan was unsafe; using a sequential pipeline")
            pipeline = run_pipeline(
                app,
                prompt,
                provider_name=active_provider,
                on_progress=on_progress,
                cancel_token=token,
                run_context=run_context,
            )
            pipeline_cost = float(getattr(pipeline, "cost", 0.0) or 0.0)
            return AutoResult(
                decision,
                pipeline.outcome,
                _format_pipeline(decision, pipeline, rerouted=True),
                pipeline.provider,
                route_in + pipeline.input_tokens,
                route_out + pipeline.output_tokens,
                route_cost + pipeline_cost,
            )
        result_cost = float(getattr(result, "cost", 0.0) or 0.0)
        return AutoResult(
            decision,
            result.outcome,
            _format_fanout(decision, result),
            result.provider,
            route_in + result.input_tokens,
            route_out + result.output_tokens,
            route_cost + result_cost,
        )

    raise ValueError("execute_auto cannot execute the chat route")
