"""Shared OpenAI-compatible tool calling logic.

Used by both OpenAIProvider and OpenRouterProvider since they share
the same chat completions API format for tool calling.
"""

import json
from typing import Callable, Optional, TYPE_CHECKING

import httpx

from .base import ToolEvent, ToolEventCallback

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
    on_usage: Optional[Callable[[tuple[int, int]], None]] = None,
    context_window: int = 128000,
    provider_preferences: Optional[dict] = None,
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

    executor = ToolExecutor(tools)

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
    total_input_tokens = 0
    total_output_tokens = 0
    budget = int(context_window * _CONTEXT_BUDGET_FRACTION)
    seen_reads: set[tuple[str, str]] = set()
    doom_streak = 0
    doom_sig = None

    def _capture_usage(data: dict) -> None:
        nonlocal total_input_tokens, total_output_tokens
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            return
        in_t = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        out_t = usage.get("completion_tokens", usage.get("output_tokens", 0))
        if isinstance(in_t, int) and isinstance(out_t, int):
            total_input_tokens += in_t
            total_output_tokens += out_t

    def _finalize_usage() -> None:
        if on_usage is not None and (total_input_tokens or total_output_tokens):
            on_usage((total_input_tokens, total_output_tokens))

    for round_num in range(max_rounds):
        # Bound the running context before every request: evict old, large tool
        # results so accumulated file reads cannot overflow the model's window.
        api_messages = _compact_messages_to_budget(
            api_messages, budget, keep_recent=_KEEP_RECENT_MESSAGES
        )
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

        choices = data.get("choices", [])
        if not choices:
            _finalize_usage()
            return "", tool_log

        message = choices[0].get("message", {})
        finish_reason = choices[0].get("finish_reason", "stop")

        tool_calls = message.get("tool_calls", [])
        content = message.get("content", "") or ""

        if not tool_calls or finish_reason != "tool_calls":
            _finalize_usage()
            return content, tool_log

        # Append the assistant message (must include tool_calls)
        api_messages.append(message)

        # Execute each tool call
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            try:
                tool_args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                tool_args = {}

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
                    result = executor.execute(tool_name, tool_args)
                    if dedup_key is not None:
                        seen_reads.add(dedup_key)
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
            f"[Stopped after {max_rounds} tool rounds without finishing -- this needs "
            f"more steps than a chat turn allows. For a multi-step build (edit, test, "
            f"commit), run /solve: it works in an isolated worktree behind your test "
            f"gate with a larger round budget.]"
        )
    return content, tool_log
