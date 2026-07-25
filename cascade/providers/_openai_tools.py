"""Shared OpenAI-compatible tool calling logic.

Used by both OpenAIProvider and OpenRouterProvider since they share
the same chat completions API format for tool calling.
"""

import json
import re
from typing import Callable, Optional, TYPE_CHECKING

import httpx

from .base import ToolEvent, ToolEventCallback
from .model_settings import settings_for
from .usage import Usage
from ..tools.permissions import PermissionAbort

if TYPE_CHECKING:
    from .base import Message
    from ..tools.schema import ToolDef


# Keep tool-call context under this fraction of the model's window, leaving
# headroom for the response. Small-context local models (e.g. a 32K Qwen)
# otherwise 400 (context-length-exceeded) after a handful of file reads.
_CONTEXT_BUDGET_FRACTION = 0.7

# Recent messages eviction must never touch -- roughly the last couple of
# tool-calling rounds -- so the model always sees its latest work in full.
_KEEP_RECENT_MESSAGES = 6

# chars-per-token heuristic, mirroring cascade.conversation.estimate_tokens.
_CHARS_PER_TOKEN = 4

# Doom-loop guard: a model repeating the exact same tool call is stalled. Nudge it
# toward a new approach on the 3rd identical call; bail on the 4th so the verified
# loop can escalate instead of spinning to the round budget. The streak counts
# repeats after the first call, so streak 2 == the 3rd identical call.
_DOOM_INTERVENE = 2
_DOOM_ABORT = 3

# Edit-run cycle guard: a model that alternates editing one file and running a
# command makes no net progress yet never issues two identical calls in a row, so
# the consecutive-identical doom guard above never trips. Count file edits per
# path and command runs per normalized prefix across the whole loop; bail once a
# single file and a single command have both churned this many times.
_EDIT_RUN_THRESHOLD = 6

# Narration guard: models flagged in the registry (cheap coders) tend to end a
# turn describing an edit instead of calling the tool. A text-only round is
# bounced back up to this many times before the text is accepted as final.
_NARRATION_LIMIT = 2
_NARRATION_NUDGE = (
    "You ended your turn without calling a tool, so no files were changed and "
    "nothing ran -- describing an edit in text does not apply it. If the task is "
    "already complete, say so plainly; otherwise call the appropriate tool now to "
    "make the change."
)
# First-person future-action language that signals an intended-but-untaken step.
# Deliberately narrow (no bare "going to" / "let's", which appear in conclusions
# like "this lets you..." / "the tests are going to pass") to avoid bouncing a
# genuine final answer.
_NARRATION_RE = re.compile(
    r"\b(i'?ll|i will|let me|i'?m going to|i am going to|now i(?:'?ll| will| need)|"
    r"next[,]? i|first[,]? i|then i'?ll|i need to|i should)\b",
    re.IGNORECASE,
)


def _looks_like_narration(content: str) -> bool:
    """Whether a tool-less round reads as an intended-but-untaken action.

    An empty round (the model just stopped) counts; a conclusive answer with no
    future-action language does not; and a clarifying question is never a
    narration -- bouncing it would pressure the model to guess instead of ask.
    """
    stripped = content.strip()
    if not stripped:
        return True
    if stripped.endswith("?"):
        return False
    return bool(_NARRATION_RE.search(stripped))

# File-mutating tools whose repeated "path" edits count toward the cycle guard.
_EDIT_TOOLS = ("write_file", "append_file", "replace_in_file")


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token count for OpenAI api_message dicts (~1 token / 4 chars).

    Counts string content plus tool-call function names and arguments (where
    large writes accumulate). Robust to the ``content: None`` that assistant
    tool-call messages carry.
    """
    chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
        for call in message.get("tool_calls") or ():
            function = call.get("function", {}) or {}
            chars += len(function.get("name", "") or "")
            chars += len(function.get("arguments", "") or "")
    return chars // _CHARS_PER_TOKEN


def _tool_names_by_call_id(messages: list[dict]) -> dict[str, str]:
    """Map each tool_call_id to the tool name that issued it."""
    names: dict[str, str] = {}
    for message in messages:
        for call in message.get("tool_calls") or ():
            call_id = call.get("id")
            if not call_id:
                continue
            function = call.get("function", {}) or {}
            names[call_id] = function.get("name") or "tool"
    return names


def _protected_indices(messages: list[dict], keep_recent: int) -> set[int]:
    """Indices eviction must never elide: system, first task, and recent tail."""
    protected: set[int] = set()
    for idx, message in enumerate(messages):
        if message.get("role") == "system":
            protected.add(idx)
            break
    for idx, message in enumerate(messages):
        if message.get("role") == "user":
            protected.add(idx)
            break
    if keep_recent > 0:
        protected.update(range(max(0, len(messages) - keep_recent), len(messages)))
    return protected


def _compact_messages_to_budget(
    messages: list[dict],
    budget: int,
    keep_recent: int = 3,
) -> list[dict]:
    """Elide old, large tool results so *messages* fits within *budget* tokens.

    Walks oldest -> newest, replacing large ``tool``-role result contents with a
    short stub and recomputing until the estimate is under budget. Never touches
    the system message, the original first user task message, or the most recent
    ``keep_recent`` messages -- those always stay full. Eliding only content (not
    removing messages) keeps every tool_call_id paired with its result, so the
    OpenAI message contract stays valid.

    Pure: returns a new list of new dicts; the input is never mutated. Idempotent
    once nothing further can be beneficially elided.
    """
    if _estimate_tokens(messages) <= budget:
        return [dict(message) for message in messages]

    names = _tool_names_by_call_id(messages)
    protected = _protected_indices(messages, keep_recent)
    result = [dict(message) for message in messages]

    for idx, message in enumerate(result):
        if _estimate_tokens(result) <= budget:
            break
        if idx in protected or message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        name = names.get(message.get("tool_call_id"), "tool")
        stub = f"[elided to fit context: {name} result, {len(content)} chars]"
        if len(stub) >= len(content):
            continue  # eliding would not save anything
        result[idx] = {**message, "content": stub}

    return result


def _read_dedup_key(tool_name: str, tool_args: dict) -> Optional[tuple[str, str]]:
    """Return a ``(tool, path)`` key for a read-style call, else ``None``.

    A call counts as a read when its name starts with ``read`` and it targets a
    concrete path argument -- the shape we can safely serve from a prior result
    instead of re-appending the full file content again.
    """
    if not tool_name.startswith("read"):
        return None
    for key in ("path", "file", "filename", "file_path"):
        value = tool_args.get(key)
        if isinstance(value, str) and value:
            return (tool_name, value)
    return None


def _invalidates_read(tool_name: str, tool_args: dict) -> Optional[str]:
    """Return the path a write/append/replace targeted, else ``None``.

    Editing a file makes any earlier read-dedup entry for that path stale: a
    later read must re-fetch fresh bytes rather than be told "[already read
    above]" pointing at the pre-edit content. Shared by the Claude and Gemini
    native loops so all three invalidate identically.
    """
    if tool_name in _EDIT_TOOLS:
        path = tool_args.get("path")
        if isinstance(path, str) and path:
            return path
    return None


def _unterminated_string(raw: str) -> bool:
    """Whether *raw* ends inside an open string literal (a truncated value)."""
    in_string = False
    escaped = False
    for ch in raw:
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            in_string = not in_string
    return in_string


def _repair_json(raw: str) -> Optional[dict]:
    """Recover a tool-argument object ONLY when doing so cannot fabricate content.

    Safe: an object whose strings are all terminated but whose closing brackets
    were cut (e.g. ``{"path": "a.py"``) -- appending ``}`` restores the true
    value. UNSAFE and refused (returns ``None``): a dangling string, which means
    a value was truncated mid-content -- closing the quote would invent a
    shorter string and, for write_file, silently write a truncated file to disk
    while reporting success. The caller reports the malformed call instead so the
    model re-issues it.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    if _unterminated_string(raw):
        return None
    opens = raw.count("{") - raw.count("}")
    brackets = raw.count("[") - raw.count("]")
    if opens <= 0 and brackets <= 0:
        return None
    candidate = raw + ("]" * max(0, brackets)) + ("}" * max(0, opens))
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _edit_run_target(tool_name: str, tool_args: dict) -> Optional[tuple[str, str]]:
    """Classify a call for the edit-run cycle guard.

    Returns ``("edit", path)`` for a file-mutating tool that names a ``path``, or
    ``("run", prefix)`` for ``run_command`` keyed by its first whitespace token (so
    ``pytest -x`` and ``pytest -v`` share one bucket), else ``None``.
    """
    if tool_name in _EDIT_TOOLS:
        path = tool_args.get("path")
        if isinstance(path, str) and path:
            return ("edit", path)
    elif tool_name == "run_command":
        command = tool_args.get("command")
        if isinstance(command, str):
            tokens = command.split()
            if tokens:
                return ("run", tokens[0])
    return None


def _hottest_over(counts: dict[str, int], threshold: int) -> Optional[str]:
    """Return the most-repeated key once it reaches *threshold*, else ``None``."""
    if not counts:
        return None
    key = max(counts, key=counts.get)
    return key if counts[key] >= threshold else None


def _api_error_text(error: object) -> str:
    """Return a useful normalized message for an in-band API error object."""
    if not isinstance(error, dict):
        return str(error or "provider request failed")
    message = str(error.get("message") or "provider request failed")
    metadata = error.get("metadata")
    if isinstance(metadata, dict) and metadata.get("error_type"):
        message = f"{message} ({metadata['error_type']})"
    return message


def openai_ask_with_tools(
    client: httpx.Client,
    url: str,
    headers: dict,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    messages: list["Message"],
    tools: dict[str, "ToolDef"],
    system: Optional[str] = None,
    max_rounds: int = 5,
    on_tool_event: ToolEventCallback = None,
    on_pending_message: Optional[Callable[[], Optional[str]]] = None,
    on_usage: Optional[Callable[[Usage], None]] = None,
    on_round_usage: Optional[Callable[[Usage], None]] = None,
    hook_runner=None,
    permissions=None,
    context_window: int = 128000,
    provider_preferences: Optional[dict] = None,
    session_id: Optional[str] = None,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> tuple[str, list[dict]]:
    """OpenAI-compatible tool calling loop.

    Args:
        client: httpx.Client instance.
        url: Chat completions endpoint URL.
        headers: Request headers with auth.
        model: Model identifier.
        temperature: Sampling temperature.
        max_tokens: Max response tokens.
        messages: Conversation history as Message dicts.
        tools: Mapping of tool_name -> ToolDef.
        system: Optional system prompt.
        max_rounds: Maximum tool-calling round trips.
        context_window: The model's real context window in tokens. Tool context
            is kept under a fraction of this before each request so small-window
            local models do not overflow as file reads accumulate.
        provider_preferences: Optional OpenRouter upstream-host routing hint,
            passed through as the request's "provider" field. None omits it.

    Returns:
        Tuple of (final_text_response, tool_calls_log).
    """
    from ..tools.executor import ToolExecutor

    executor = ToolExecutor(tools, hook_runner=hook_runner, permissions=permissions)

    # Build OpenAI tool definitions
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
            },
        }
        for td in tools.values()
    ]

    api_messages = []
    if system:
        api_messages.append({"role": "system", "content": system})
    api_messages.extend(
        {"role": m["role"], "content": m["content"]}
        for m in messages
    )

    tool_log = []
    content = ""
    total_usage = Usage()
    round_usage = None
    budget = int(context_window * _CONTEXT_BUDGET_FRACTION)
    # dedup_key -> the tool_call_id whose result holds that read. When budget
    # compaction later ELIDES that result, the key is dropped so a repeat read
    # re-fetches the file instead of being told "[already read above]" while the
    # content it points at is gone (the model could then neither see nor re-read
    # it -- a real hole in long loops).
    seen_reads: dict[tuple[str, str], str] = {}
    doom_streak = 0
    doom_sig = None
    edit_counts: dict[str, int] = {}
    run_counts: dict[str, int] = {}
    narration_streak = 0

    # Per-model tuning: cheap coders want near-0 sampling and narration bounces.
    settings = settings_for(model)
    if settings.temperature is not None:
        temperature = settings.temperature

    def _capture_usage(data: dict) -> None:
        nonlocal total_usage, round_usage
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            return
        parsed = Usage.from_openai(usage)
        if parsed.total or parsed.cost:
            total_usage = total_usage.add(parsed)
            round_usage = parsed

    def _finalize_usage() -> None:
        if on_usage is not None and (total_usage.total or total_usage.cost):
            on_usage(total_usage)
        if on_round_usage is not None and round_usage is not None:
            on_round_usage(round_usage)

    def _checkpoint() -> None:
        if check_cancelled is not None:
            check_cancelled()

    for round_num in range(max_rounds):
        _checkpoint()
        # Land a prompt the user queued mid-turn at this round boundary
        # (codex-style), so a follow-up steers the very next model call instead
        # of waiting for the whole turn. A 'tool' message may precede it; a
        # following 'user' message is valid in the chat format.
        if on_pending_message is not None:
            pending = on_pending_message()
            if pending:
                api_messages.append({"role": "user", "content": pending})
        # Bound the running context before every request: evict old, large tool
        # results so accumulated file reads cannot overflow the model's window.
        _before = api_messages
        api_messages = _compact_messages_to_budget(
            api_messages, budget, keep_recent=_KEEP_RECENT_MESSAGES
        )
        # A read whose result was just elided must become re-readable: drop its
        # dedup key so the model can fetch the file again rather than be pointed
        # at content that no longer exists. (_compact returns a same-length list,
        # so a tool message whose content changed was elided.)
        if seen_reads and len(_before) == len(api_messages):
            elided_ids = {
                new.get("tool_call_id")
                for old, new in zip(_before, api_messages)
                if old.get("role") == "tool" and old.get("content") != new.get("content")
            }
            if elided_ids:
                seen_reads = {
                    key: cid for key, cid in seen_reads.items() if cid not in elided_ids
                }
        payload = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "tools": tool_defs,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if provider_preferences:
            payload["provider"] = provider_preferences
        # Pin the whole conversation to one upstream host so its prompt cache stays
        # warm across the tool loop's growing rounds (OpenRouter sticky sessions).
        if session_id:
            payload["session_id"] = session_id

        try:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            _capture_usage(data)
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(str(exc)) from exc
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

        _checkpoint()

        # OpenRouter and some OpenAI-compatible providers can commit an HTTP 200
        # before generation fails, then report the failure in the response body.
        # Treating that as an empty assistant message lets truncated agent work
        # masquerade as a normal round.
        if data.get("error"):
            _finalize_usage()
            raise RuntimeError(_api_error_text(data["error"]))

        choices = data.get("choices", [])
        if not choices:
            _finalize_usage()
            return "", tool_log

        choice_error = choices[0].get("error")
        if choices[0].get("finish_reason") == "error" or choice_error:
            _finalize_usage()
            raise RuntimeError(_api_error_text(choice_error))

        message = choices[0].get("message", {})
        finish_reason = choices[0].get("finish_reason", "stop")

        tool_calls = message.get("tool_calls", [])
        content = message.get("content", "") or ""

        if not tool_calls:
            # Narration bounce: a registry-flagged model that ends a turn
            # describing an edit (or just stops) instead of calling a tool is
            # pushed to actually act, rather than the run silently ending with
            # no changes. Bounded so a genuine "done" still returns.
            if (
                settings.nudge_on_narration
                and narration_streak < _NARRATION_LIMIT
                and round_num < max_rounds - 1
                and _looks_like_narration(content)
            ):
                narration_streak += 1
                if content.strip():
                    api_messages.append({"role": "assistant", "content": content})
                api_messages.append({"role": "user", "content": _NARRATION_NUDGE})
                continue
            _finalize_usage()
            if narration_streak > 0:
                # Bounced but the model still did not act -- flag that its
                # description was not applied, so it doesn't read as success.
                note = "[no changes made -- described the edit but did not perform it]"
                return ((content.rstrip() + "\n\n" + note) if content.strip() else note), tool_log
            return content, tool_log

        narration_streak = 0

        # Tool calls present -> execute them REGARDLESS of finish_reason. The old
        # `finish_reason != "tool_calls"` gate silently dropped calls from compat
        # hosts (vLLM-class: local Kimi/GLM and many OpenRouter endpoints) that
        # report finish_reason "stop" alongside tool calls, or "length" when the
        # output limit truncates the final call -- which is exactly the "produced
        # no changes, no error" failure. Truncated arguments are recovered or
        # reported per-call below rather than executed as empty.

        # Append the assistant message (must include tool_calls)
        api_messages.append(message)

        # Execute each tool call
        for tc in tool_calls:
            _checkpoint()
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            raw_args = fn.get("arguments", "") or "{}"
            try:
                tool_args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                repaired = _repair_json(raw_args)
                if repaired is not None:
                    tool_args = repaired
                else:
                    # Repeated identical broken calls (a write that truncates the
                    # same way each round) must trip the stall guard rather than
                    # burn every round re-issuing it, so account for them here --
                    # the parse failed, so the doom signature keys on raw_args.
                    sig = (tool_name, raw_args)
                    doom_streak = doom_streak + 1 if sig == doom_sig else 0
                    doom_sig = sig
                    if doom_streak >= _DOOM_ABORT:
                        _finalize_usage()
                        note = (
                            f"[stalled: {tool_name} arguments were truncated "
                            f"{doom_streak + 1}x with no progress -- handing off.]"
                        )
                        return (content if content.strip() else note), tool_log
                    # Never fall back to {}: running a tool with empty arguments
                    # looks like a successful no-op. Report the malformed call so
                    # the model re-issues it (common when a large write_file payload
                    # is truncated at the output limit).
                    err = (
                        f"[tool-call error] arguments were not valid JSON: {exc}. "
                        f"Re-issue {tool_name} with complete, valid JSON arguments "
                        f"(make the payload smaller if it was cut off)."
                    )
                    tool_log.append({"tool": tool_name, "input": raw_args, "output": err})
                    api_messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": err}
                    )
                    continue

            # Doom-loop guard: the same tool + same args over and over is a stall.
            # A different call in between resets it (so read -> edit -> read is fine).
            sig = (tool_name, json.dumps(tool_args, sort_keys=True, default=str))
            if sig == doom_sig:
                doom_streak += 1
            else:
                doom_streak, doom_sig = 0, sig
            if doom_streak >= _DOOM_ABORT:
                _finalize_usage()
                note = (
                    f"[stalled: called {tool_name} with identical arguments "
                    f"{doom_streak + 1}x with no progress -- handing off.]"
                )
                return (content if content.strip() else note), tool_log

            # Edit-run cycle guard: the doom guard above only catches back-to-back
            # identical calls, so a model thrashing edit -> run -> edit slips past
            # it. Tally edits per file and runs per command prefix across the whole
            # loop; once one file and one command have both churned past the
            # threshold, bail so the verified loop can escalate instead of spinning.
            target = _edit_run_target(tool_name, tool_args)
            if target is not None:
                kind, key = target
                counts = edit_counts if kind == "edit" else run_counts
                counts[key] = counts.get(key, 0) + 1
                stuck_file = _hottest_over(edit_counts, _EDIT_RUN_THRESHOLD)
                if stuck_file and _hottest_over(run_counts, _EDIT_RUN_THRESHOLD):
                    _finalize_usage()
                    note = (
                        f"[stalled: edit-run cycle on {stuck_file} without "
                        f"progress -- handing off.]"
                    )
                    return (content if content.strip() else note), tool_log

            if on_tool_event:
                on_tool_event(ToolEvent(
                    kind="tool_start",
                    tool_name=tool_name,
                    round_num=round_num,
                    max_rounds=max_rounds,
                    tool_input=tool_args,
                ))

            if doom_streak == _DOOM_INTERVENE:
                # 3rd identical call: don't run it again -- nudge to change tack.
                result = (
                    f"You have called {tool_name} with identical arguments "
                    f"{doom_streak + 1} times with the same result. Stop repeating "
                    f"it and try a different approach."
                )
            else:
                # Read de-duplication: serve a repeat read of an already-read path
                # from a short stub instead of re-appending the full file content.
                dedup_key = _read_dedup_key(tool_name, tool_args)
                if dedup_key is not None and dedup_key in seen_reads:
                    result = f"[already read above: {dedup_key[1]}]"
                else:
                    try:
                        result = executor.execute(tool_name, tool_args)
                    except PermissionAbort as exc:
                        _finalize_usage()
                        note = f"\n\n[stopped: {exc}]"
                        return (content + note).strip(), tool_log
                    if dedup_key is not None:
                        seen_reads[dedup_key] = tc["id"]
                    else:
                        # A successful edit makes any cached read of that path
                        # stale: drop its dedup keys so the next read re-fetches
                        # fresh content instead of "[already read above]".
                        edited = _invalidates_read(tool_name, tool_args)
                        if edited is not None and seen_reads:
                            seen_reads = {
                                key: cid
                                for key, cid in seen_reads.items()
                                if key[1] != edited
                            }
            _checkpoint()
            tool_log.append({
                "tool": tool_name,
                "input": tool_args,
                "output": result,
            })

            if on_tool_event:
                on_tool_event(ToolEvent(
                    kind="tool_done",
                    tool_name=tool_name,
                    round_num=round_num,
                    max_rounds=max_rounds,
                    tool_input=tool_args,
                    tool_output=result,
                ))

            api_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    _finalize_usage()
    if not content.strip():
        # The loop spent its whole round budget still tool-calling and never
        # produced a final answer. Returning "" here is what looked like the model
        # "just stopping" -- surface it instead, and point at the right tool.
        content = (
            f"[Paused after {max_rounds} tool steps -- send another message to keep "
            f"going, or /solve if you want it done in an isolated, test-verified "
            f"worktree.]"
        )
    return content, tool_log
