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
from .worktree import WorktreeManager


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
    history_hint: str = ""


@dataclass(frozen=True)
class AutoResult:
    decision: RouteDecision
    outcome: RunOutcome
    text: str
    execution_provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    # Per-provider breakdown carried up from the underlying Solve/Pipeline/Fanout
    # result (plus the router's own call) so an escalation's tokens/cost are
    # credited to the model that incurred them, not lumped under the base
    # provider. (provider, input_tokens, output_tokens) and (provider, cost).
    tokens_by_provider: tuple[tuple[str, int, int], ...] = ()
    cost_by_provider: tuple[tuple[str, float], ...] = ()
    worktree_path: str = ""
    changed_files: tuple[str, ...] = ()
    diff_stat: str = ""
    iterations: int = 0
    verification_kind: str = ""
    models_used: tuple[str, ...] = ()


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
You route software-assistant requests. Ordinary prose is the primary interface:
the user should not need to know or remember workflow commands. Choose the
smallest workflow that can complete the whole request safely:

- chat: conversation, ideation, or a question that needs no repository access.
- recon: read/search/trace/review the repository without changing any file.
- solve: one focused code change that can be verified in one worktree.
- pipeline: multiple dependent code changes that must build on one another.
- fanout: multiple genuinely independent code changes with disjoint file ownership.

Choose pipeline when the deliverables depend on earlier investigation or edits.
Choose fanout when two or more substantial deliverables have credible disjoint
file ownership and can be verified independently. Prefer solve when coordination
would add no value. Never infer a code edit from explanation or review alone.

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

_FAST_EDIT_RE = re.compile(
    r"\b(typo|one[- ]line|tiny|small localized|formatting only)\b",
    re.IGNORECASE,
)
_COMPLEX_EDIT_RE = re.compile(
    r"(?:"
    r"\b(?:across|throughout|end[- ]to[- ]end|multiple|several|all of|"
    r"work through|full pass|whole project|each provider|every provider|"
    r"independent|in parallel|fan out|subagents?)\b"
    r"|(?:^|\n)\s*(?:[-*]|\d+[.)])\s+"
    r"|\b(?:and then|followed by|after that)\b"
    r")",
    re.IGNORECASE,
)
# Precision-first LOCAL router -- a handful of ALLOWLIST rules that fire only on
# unambiguous imperatives and abstain (return None) on everything else, deferring
# to the model router. This deliberately favors precision over recall (the
# roadmap's "allowlist-known-transparent > denylist-known-bad" lesson): it must
# never confidently mis-route. The embedding tier (Phase 2) grows recall from
# this safe floor.

# An edit imperative: a core change verb governing a concrete object
# (determiner + word). Anchored at the start, so a question ("should we add...")
# or a buried mention ("the tests add coverage") cannot match, and a bare
# pronoun object ("fix it") fails the determiner requirement -> abstains.
_SOLVE_ALLOW_RE = re.compile(
    r"^(?:please )?"
    r"(fix|add|remove|delete|rename|implement|create|update|change|refactor|"
    r"replace|wire|build|make|extract|move|split|revert|bump|upgrade|migrate|"
    r"write|patch|correct|repair|introduce|enable|disable|convert|rewrite) "
    r"(the|a|an|this|that|these|those|my|our|its|all|each|every|another|some) "
    r"\S+",
    re.IGNORECASE,
)
# A read imperative naming a repository object -> recon. A qualifier or two may
# sit before the object ("review the AUTH module"), but the object set is
# deliberately code-specific and excludes generic nouns like "function"/"class",
# so general knowledge ("explain how the map function works") abstains rather
# than spawning a read-only pass.
_RECON_ALLOW_RE = re.compile(
    r"^(?:please )?"
    r"(read|review|inspect|explain|trace|audit|examine|summari[sz]e|describe) "
    r"(?:the |this |that |our |my |its )?"
    r"(?:[\w-]+ ){0,2}"
    r"(repo|repository|codebase|code|module|modules|implementation|file|files|"
    r"config|configuration|schema|package|packages|endpoint|endpoints|handler|"
    r"handlers|component|components|widget|widgets|screen|screens|provider|"
    r"providers|diff|commit|commits|test suite|tests?)\b",
    re.IGNORECASE,
)
# The WHOLE prompt is a greeting/acknowledgement -> chat. fullmatch means a
# greeting PREFIX on an action ("hey, continue fixing it") does not match.
_GREETING_RE = re.compile(
    r"\s*(hi|hey|hello|thanks|thank you|thx|ty|yo|sup|ok|okay|cool|nice|great|"
    r"perfect|awesome|good (?:morning|afternoon|evening))"
    r"[\s,!.]*(that worked|it works|works|thanks|perfect)?[\s.!]*",
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


def _safe_fallback(reason: str) -> RouteDecision:
    """When the model router is unavailable or fails, fall back to CHAT.

    Chat has file + shell tools and can edit/run directly, so an unrouted prompt
    still gets handled -- without the old lexical heuristic's habit of spawning a
    worktree solve for a prompt it only pattern-matched as an edit.
    """
    return RouteDecision(
        workflow=WorkflowKind.CHAT,
        reason=reason,
        confidence=0.5,
        router_provider="fallback",
    )

def _routing_history_hint(app) -> str:
    """A compact, anonymous tie-breaker derived from this installation's runs."""
    ledger = getattr(app, "run_ledger", None)
    if ledger is None:
        return ""
    try:
        summary = ledger.routing_summary(limit=100)
    except Exception:
        return ""
    parts = []
    for workflow in ("solve", "pipeline", "fanout"):
        bucket = summary.get(workflow)
        if not bucket or bucket.get("runs", 0) < 3:
            continue
        runs = int(bucket["runs"])
        succeeded = int(bucket["succeeded"])
        parts.append(f"{workflow} {succeeded}/{runs} succeeded")
    return "; ".join(parts)


def _local_route(prompt: str, mode: str) -> Optional[RouteDecision]:
    """A zero-cost first tier: a confident decision only on an unambiguous
    imperative, else ``None`` to defer to the model router.

    Three allowlist rules -- an edit imperative governing a concrete object ->
    solve, a repo-scoped read imperative -> recon, a whole-prompt greeting ->
    chat. Everything else (questions, referential or conversational phrasing, any
    multi-part or ambiguous ask) abstains. Precision over recall by design: it
    must never confidently mis-route, so recall is intentionally low until the
    embedding tier lands. ``mode`` suppresses solve routing in the think-first
    modes, where an edit-shaped ask is more likely ideation than a command.
    """
    text = prompt.strip()
    if not text or text.endswith("?"):
        return None

    def _decide(workflow: WorkflowKind, reason: str, tier: str = "bulk") -> RouteDecision:
        return RouteDecision(
            workflow=workflow,
            reason=f"local rule: {reason}",
            confidence=0.9,
            worker_tier=tier,
            router_provider="local",
        )

    if _GREETING_RE.fullmatch(text):
        return _decide(WorkflowKind.CHAT, "a bare greeting")
    if _RECON_ALLOW_RE.match(text):
        return _decide(WorkflowKind.RECON, "imperative repository inspection")
    if (
        mode not in ("plan", "design")
        and _SOLVE_ALLOW_RE.match(text)
        and not _COMPLEX_EDIT_RE.search(text)
    ):
        tier = "fast" if _FAST_EDIT_RE.search(text) else "bulk"
        return _decide(WorkflowKind.SOLVE, "imperative edit command", tier)
    return None


def select_workflow(
    app,
    prompt: str,
    mode: str,
    cancel_token: Optional[CancellationToken] = None,
) -> RouteDecision:
    """Route a prompt to a workflow: a zero-cost local rule when it is
    unambiguous, otherwise the configured cheap model, failing conservatively."""
    if cancel_token is not None:
        cancel_token.checkpoint()
    config = app.config.get_orchestration_config()
    history_hint = _routing_history_hint(app)
    if config.get("local_router", True):
        local = _local_route(prompt, mode)
        if local is not None:
            return local
    provider_name = config["router_provider"]
    model = config["router_model"]
    router = _lane_provider(
        app,
        provider_name,
        model,
        config["provider_preferences"],
    )
    if router is None:
        return _safe_fallback("routing model unavailable; handled as chat")

    try:
        request = f"Current Cascade mode: {mode}\n\nUser request:\n{prompt}"
        if history_hint:
            request += (
                "\n\nRecent local outcomes (weak tie-breaker only; never override "
                f"the request): {history_hint}"
            )
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
            history_hint=history_hint,
        )
    except RunCancelled:
        raise
    except Exception:
        if cancel_token is not None:
            cancel_token.checkpoint()
        return _safe_fallback("routing model failed; handled as chat")


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


def _route_token_attribution(
    decision: RouteDecision,
) -> "tuple[tuple[str, int, int], ...]":
    """The router call's own tokens, attributed to the provider that ran it.

    Emitted only when the route actually spent tokens, so the zero-cost local /
    heuristic / fallback routers add no phantom entry to the breakdown.
    """
    if decision.input_tokens or decision.output_tokens:
        return (
            (decision.router_provider, decision.input_tokens, decision.output_tokens),
        )
    return ()


def _route_cost_attribution(
    decision: RouteDecision,
) -> "tuple[tuple[str, float], ...]":
    """The router call's own cost, attributed to the provider that ran it."""
    if decision.cost:
        return ((decision.router_provider, decision.cost),)
    return ()


def _merge_token_breakdown(
    *groups: "tuple[tuple[str, int, int], ...]",
) -> "tuple[tuple[str, int, int], ...]":
    """Sum per-provider token tuples across groups, preserving first-seen order."""
    merged: dict[str, list[int]] = {}
    for group in groups:
        for label, ins, outs in group:
            bucket = merged.setdefault(label, [0, 0])
            bucket[0] += ins
            bucket[1] += outs
    return tuple((label, ins, outs) for label, (ins, outs) in merged.items())


def _merge_cost_breakdown(
    *groups: "tuple[tuple[str, float], ...]",
) -> "tuple[tuple[str, float], ...]":
    """Sum per-provider cost tuples across groups, preserving first-seen order."""
    merged: dict[str, float] = {}
    for group in groups:
        for label, cost in group:
            merged[label] = merged.get(label, 0.0) + cost
    return tuple(merged.items())


def _auto_breakdowns(
    decision: RouteDecision, *results
) -> "tuple[tuple[tuple[str, int, int], ...], tuple[tuple[str, float], ...]]":
    """Merged per-provider (tokens, cost) for the route call plus each result.

    ``getattr`` guards let test doubles omit the breakdown fields; real
    Solve/Pipeline/Fanout results always carry them.
    """
    token_groups = [_route_token_attribution(decision)]
    cost_groups = [_route_cost_attribution(decision)]
    for result in results:
        token_groups.append(tuple(getattr(result, "tokens_by_provider", ()) or ()))
        cost_groups.append(tuple(getattr(result, "cost_by_provider", ()) or ()))
    return (
        _merge_token_breakdown(*token_groups),
        _merge_cost_breakdown(*cost_groups),
    )


def execute_auto(
    app,
    prompt: str,
    active_provider: str,
    decision: RouteDecision,
    *,
    mode: str = "",
    context: str = "",
    on_progress: ProgressCallback = None,
    on_tool_event=None,
    cancel_token: Optional[CancellationToken] = None,
    run_context: Optional[RunContext] = None,
) -> AutoResult:
    """Run an automatic lane with one consistent orchestration hook envelope."""
    from ..hooks import HookContext, HookEvent

    runner = getattr(app, "hook_runner", None)
    if runner is None:
        return _execute_auto_impl(
            app,
            prompt,
            active_provider,
            decision,
            mode=mode,
            context=context,
            on_progress=on_progress,
            on_tool_event=on_tool_event,
            cancel_token=cancel_token,
            run_context=run_context,
        )

    start = runner.emit(
        HookEvent.WORKFLOW_START,
        HookContext(
            event=HookEvent.WORKFLOW_START.value,
            provider=active_provider,
            mode=mode,
            workflow=decision.workflow.value,
            prompt=prompt,
        ),
    )
    if start is not None:
        if start.block:
            return AutoResult(
                decision=decision,
                outcome=RunOutcome.BLOCKED,
                text=f"Workflow blocked by hook: {start.reason}",
                execution_provider=active_provider,
                input_tokens=decision.input_tokens,
                output_tokens=decision.output_tokens,
                cost=decision.cost,
            )
        if start.transformed_value is not None:
            if not isinstance(start.transformed_value, str):
                return AutoResult(
                    decision=decision,
                    outcome=RunOutcome.BLOCKED,
                    text="Workflow hook returned an invalid prompt (expected text)",
                    execution_provider=active_provider,
                    input_tokens=decision.input_tokens,
                    output_tokens=decision.output_tokens,
                    cost=decision.cost,
                )
            prompt = start.transformed_value

    try:
        result = _execute_auto_impl(
            app,
            prompt,
            active_provider,
            decision,
            mode=mode,
            context=context,
            on_progress=on_progress,
            on_tool_event=on_tool_event,
            cancel_token=cancel_token,
            run_context=run_context,
        )
    except Exception as exc:
        runner.emit(
            HookEvent.ON_ERROR,
            HookContext(
                event=HookEvent.ON_ERROR.value,
                provider=active_provider,
                mode=mode,
                workflow=decision.workflow.value,
                prompt=prompt,
                error=str(exc),
            ),
        )
        raise

    end = runner.emit(
        HookEvent.WORKFLOW_END,
        HookContext(
            event=HookEvent.WORKFLOW_END.value,
            provider=result.execution_provider,
            mode=mode,
            workflow=decision.workflow.value,
            prompt=prompt,
            response=result.text,
            metadata=(
                ("outcome", result.outcome.value),
                ("input_tokens", result.input_tokens),
                ("output_tokens", result.output_tokens),
                ("cost", result.cost),
            ),
        ),
    )
    if end is not None:
        if end.block:
            return replace(
                result,
                outcome=RunOutcome.BLOCKED,
                text=f"Workflow result blocked by hook: {end.reason}",
            )
        if end.transformed_value is not None:
            if isinstance(end.transformed_value, str):
                result = replace(result, text=end.transformed_value)
            else:
                result = replace(
                    result,
                    outcome=RunOutcome.BLOCKED,
                    text="Workflow end hook returned an invalid response (expected text)",
                )
    return result


def _execute_auto_impl(
    app,
    prompt: str,
    active_provider: str,
    decision: RouteDecision,
    *,
    mode: str = "",
    context: str = "",
    on_progress: ProgressCallback = None,
    on_tool_event=None,
    cancel_token: Optional[CancellationToken] = None,
    run_context: Optional[RunContext] = None,
) -> AutoResult:
    """Execute a non-chat route selected by :func:`select_workflow`.

    ``context`` is an optional bounded conversation digest (from
    :func:`cascade.conversation.build_lane_context`) fed into the chosen lane so
    a referential prompt -- "fix the errors codex found" -- can resolve its
    referent; the focused lane otherwise runs the bare prompt with no history.
    """
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
        verify_mode = mode == "test"
        disposable_provider = provider is not None
        if provider is None:
            # A direct-API active provider can still be constrained to read-only
            # tools. A CLI proxy normally cannot -- its native tool set is not
            # controlled by WorkspaceTools -- EXCEPT in test mode, where the
            # _force_repo_write override below hands it a writable sandbox to
            # actually run the project's checks, so it may serve as recon there.
            candidate = app.providers.get(active_provider)
            candidate_is_cli_proxy = candidate is not None and (
                getattr(candidate, "_use_cli_proxy", False)
                or getattr(candidate, "_use_oauth_cli", False)
            )
            if candidate is not None and (verify_mode or not candidate_is_cli_proxy):
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
            # prompt that permits running checks but not editing source. Because
            # that run_command (and, for a CLI proxy, its writable sandbox) must
            # never touch the user's real checkout, test-mode recon builds
            # against a throwaway worktree at HEAD -- the same isolation /solve
            # uses -- so run_command's isolation invariant holds. Read-only recon
            # inspects the repository in place.
            recon_root = os.getcwd()
            verify_manager = None
            worktree_scope = nullcontext()
            force_write_present = hasattr(provider, "_force_repo_write")
            force_write_prev = getattr(provider, "_force_repo_write", None)
            # The try/finally spans every verify-mode side effect (the writable
            # sandbox flag and the throwaway worktree) so both are always undone,
            # even if setup fails before the model call.
            try:
                if verify_mode:
                    verify_manager = WorktreeManager()
                    recon_root = verify_manager.prepare("_recon").path
                    # A CLI proxy (codex) drives its own sandbox in its cwd, not
                    # our WorkspaceTools, so aim it at the worktree and force its
                    # writable sandbox. Harmless on direct-API providers, which
                    # have neither.
                    working_directory = getattr(provider, "working_directory", None)
                    if callable(working_directory):
                        worktree_scope = working_directory(recon_root)
                    setattr(provider, "_force_repo_write", True)
                # A verify recon actually runs the project's suite, which the 120s
                # default can easily outlast; a read-only recon runs no commands.
                command_timeout = 600.0 if verify_mode else 120.0
                ws = WorkspaceTools(
                    recon_root, command_timeout=command_timeout, cancel_token=token,
                )
                recon_tools = ws.build_verify() if verify_mode else ws.build_read_only()
                recon_system = (
                    _RECON_VERIFY_SYSTEM if verify_mode else _RECON_READONLY_SYSTEM
                )
                recon_input = f"{context}\n\n{prompt}" if context else prompt
                with scope, callback_scope, worktree_scope:
                    response, _log = provider.ask_with_tools(
                        [{"role": "user", "content": recon_input}],
                        recon_tools,
                        system=recon_system,
                        max_rounds=config["recon_max_rounds"],
                        on_tool_event=on_tool_event,
                    )
            finally:
                if verify_mode:
                    # Restore the shared provider: never leave a writable-sandbox
                    # flag on the user's interactive provider after recon.
                    if force_write_present:
                        setattr(provider, "_force_repo_write", force_write_prev)
                    else:
                        try:
                            delattr(provider, "_force_repo_write")
                        except AttributeError:
                            pass
                    if verify_manager is not None:
                        verify_manager.cleanup()
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
                models_used=(str(provider.config.model),),
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
            context=context,
        )
        result_cost = float(getattr(result, "cost", 0.0) or 0.0)
        tokens_bd, cost_bd = _auto_breakdowns(decision, result)
        return AutoResult(
            decision,
            result.outcome,
            _format_solve(decision, result),
            result.provider,
            route_in + result.input_tokens,
            route_out + result.output_tokens,
            route_cost + result_cost,
            tokens_bd,
            cost_bd,
            worktree_path=result.worktree_path,
            changed_files=tuple(getattr(result, "changed_files", ()) or ()),
            diff_stat=getattr(result, "diff_stat", ""),
            iterations=getattr(result, "iterations", 0),
            verification_kind=getattr(result, "verification_kind", ""),
            models_used=tuple(getattr(result, "models_used", ()) or ()),
        )

    if decision.workflow == WorkflowKind.PIPELINE:
        result = run_pipeline(
            app,
            prompt,
            provider_name=active_provider,
            on_progress=on_progress,
            cancel_token=token,
            run_context=run_context,
            context=context,
        )
        result_cost = float(getattr(result, "cost", 0.0) or 0.0)
        tokens_bd, cost_bd = _auto_breakdowns(decision, result)
        return AutoResult(
            decision,
            result.outcome,
            _format_pipeline(decision, result),
            result.provider,
            route_in + result.input_tokens,
            route_out + result.output_tokens,
            route_cost + result_cost,
            tokens_bd,
            cost_bd,
            worktree_path=result.worktree_path,
            changed_files=tuple(getattr(result, "changed_files", ()) or ()),
            diff_stat=getattr(result, "diff_stat", ""),
            iterations=sum(getattr(step, "iterations", 0) for step in result.steps),
            verification_kind=getattr(result, "verification_kind", ""),
            models_used=tuple(
                model
                for step in result.steps
                for model in (getattr(step, "models_used", ()) or ())
            ),
        )

    if decision.workflow == WorkflowKind.FANOUT:
        result = run_fanout(
            app,
            prompt,
            provider_name=active_provider,
            on_progress=on_progress,
            cancel_token=token,
            run_context=run_context,
            context=context,
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
                context=context,
            )
            pipeline_cost = float(getattr(pipeline, "cost", 0.0) or 0.0)
            # The blocked fanout still made a real director planning call; fold
            # its own tokens/cost in so the reroute does not erase that spend.
            # Its cost is taken from the per-provider breakdown (the authoritative
            # record of what each model billed) rather than the flat ``cost``
            # field, keeping the AutoResult's flat cost equal to its breakdown.
            blocked_in = getattr(result, "input_tokens", 0) or 0
            blocked_out = getattr(result, "output_tokens", 0) or 0
            blocked_cost = sum(
                c for _, c in (getattr(result, "cost_by_provider", ()) or ())
            )
            tokens_bd, cost_bd = _auto_breakdowns(decision, result, pipeline)
            return AutoResult(
                decision,
                pipeline.outcome,
                _format_pipeline(decision, pipeline, rerouted=True),
                pipeline.provider,
                route_in + blocked_in + pipeline.input_tokens,
                route_out + blocked_out + pipeline.output_tokens,
                route_cost + blocked_cost + pipeline_cost,
                tokens_bd,
                cost_bd,
                worktree_path=pipeline.worktree_path,
                changed_files=tuple(getattr(pipeline, "changed_files", ()) or ()),
                diff_stat=getattr(pipeline, "diff_stat", ""),
                iterations=sum(
                    getattr(step, "iterations", 0) for step in pipeline.steps
                ),
                verification_kind=getattr(pipeline, "verification_kind", ""),
                models_used=tuple(
                    model
                    for step in pipeline.steps
                    for model in (getattr(step, "models_used", ()) or ())
                ),
            )
        result_cost = float(getattr(result, "cost", 0.0) or 0.0)
        tokens_bd, cost_bd = _auto_breakdowns(decision, result)
        return AutoResult(
            decision,
            result.outcome,
            _format_fanout(decision, result),
            result.provider,
            route_in + result.input_tokens,
            route_out + result.output_tokens,
            route_cost + result_cost,
            tokens_bd,
            cost_bd,
            worktree_path=result.worktree_path,
            changed_files=tuple(getattr(result, "changed_files", ()) or ()),
            diff_stat=getattr(result, "diff_stat", ""),
            iterations=sum(getattr(sub, "iterations", 0) for sub in result.subs),
            verification_kind=getattr(result, "verification_kind", ""),
            models_used=tuple(
                model
                for sub in result.subs
                for model in (getattr(sub, "models_used", ()) or ())
            ),
        )

    raise ValueError("execute_auto cannot execute the chat route")
