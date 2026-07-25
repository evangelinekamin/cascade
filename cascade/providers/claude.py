"""Anthropic Claude provider implementation."""

from typing import Optional, Iterator, TYPE_CHECKING
import json
import os
import shutil
import httpx
from dataclasses import replace
from .base import BaseProvider, ProviderConfig, Message, ToolEvent, ToolEventCallback
from ._cli_proxy import CLIProxyConfig, ClaudeEventHandler, stream_cli_proxy
from ._openai_tools import (
    _CHARS_PER_TOKEN,
    _CONTEXT_BUDGET_FRACTION,
    _KEEP_RECENT_MESSAGES,
    _invalidates_read,
    _read_dedup_key,
)
from .registry import register_provider
from .usage import Usage
from ..context.budget import window_for

if TYPE_CHECKING:
    from ..tools.schema import ToolDef


# Opus 4.7+, Sonnet 5, and the Fable/Mythos family reject sampling params
# (temperature / top_p / top_k) with a 400. Older models still accept them.
_NO_SAMPLING_TAGS = ("opus-4-7", "opus-4-8", "sonnet-5", "fable-5", "mythos")


def _accepts_sampling(model: str) -> bool:
    return not any(tag in model for tag in _NO_SAMPLING_TAGS)


class _AccumulatingClaudeEventHandler(ClaudeEventHandler):
    """Sum output across ``claude -p``'s internal turns.

    ``claude -p`` runs a whole agentic solve in one subprocess, emitting a
    message_start/message_delta pair per internal turn. The base handler keeps
    only the latest turn's usage, so an 8k-line solve reported a single turn's
    output. Bank each finished turn: ``total_usage`` is the accumulated spend,
    while the base ``last_usage`` stays the final-turn context anchor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.total_usage: Optional[Usage] = None
        self._turn_usage: Optional[Usage] = None

    def on_json_event(self, event: dict) -> Iterator[tuple[str, str]]:
        inner = event.get("event") if event.get("type") == "stream_event" else None
        inner_type = inner.get("type") if isinstance(inner, dict) else None
        if inner_type == "message_start":
            self.bank_final_turn()  # a new turn begins: commit the finished one
        yield from super().on_json_event(event)
        # message_start seeds the turn's input; message_delta finalizes its
        # output. The result event (which overwrites last_usage with the final
        # turn only) is deliberately not folded in a second time.
        if inner_type in ("message_start", "message_delta"):
            self._turn_usage = self.last_usage

    def bank_final_turn(self) -> None:
        """Fold the in-flight turn into the accumulated total."""
        if self._turn_usage is not None:
            self.total_usage = (
                self._turn_usage
                if self.total_usage is None
                else self.total_usage.add(self._turn_usage)
            )
            self._turn_usage = None


def _estimate_anthropic_tokens(messages: list[dict]) -> int:
    """Rough token count (~1 token / 4 chars) for Anthropic api_message dicts.

    Counts string content, tool_use inputs, and tool_result content -- where
    accumulated file reads pile up. Mirrors the OpenAI-loop estimator.
    """
    chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "text":
                    chars += len(block.get("text") or "")
                elif kind == "tool_use":
                    chars += len(json.dumps(block.get("input") or {}, default=str))
                elif kind == "tool_result" and isinstance(block.get("content"), str):
                    chars += len(block["content"])
    return chars // _CHARS_PER_TOKEN


def _compact_anthropic_tool_results(
    messages: list[dict], budget: int, keep_recent: int,
) -> list[dict]:
    """Elide old, large tool_result blocks to a stub so *messages* fits *budget*.

    Only the inner content string of a tool_result is replaced -- the block, its
    ``tool_use_id``, and every tool_use stay in place, so the Anthropic
    tool_use/tool_result pairing contract is never broken. Protects the first
    user turn (the task) and the most recent *keep_recent* messages. Pure:
    returns new dicts on eviction; the input list is never mutated.
    """
    if _estimate_anthropic_tokens(messages) <= budget:
        return messages
    protected: set[int] = set()
    for idx, message in enumerate(messages):
        if message.get("role") == "user":
            protected.add(idx)
            break
    if keep_recent > 0:
        protected.update(range(max(0, len(messages) - keep_recent), len(messages)))

    result = [dict(message) for message in messages]
    for idx, message in enumerate(result):
        if _estimate_anthropic_tokens(result) <= budget:
            break
        if idx in protected or not isinstance(message.get("content"), list):
            continue
        new_blocks: list = []
        changed = False
        for block in message["content"]:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and isinstance(block.get("content"), str)
            ):
                original = block["content"]
                stub = f"[elided to fit context: tool result, {len(original)} chars]"
                if len(stub) < len(original):
                    new_blocks.append({**block, "content": stub})
                    changed = True
                    continue
            new_blocks.append(block)
        if changed:
            result[idx] = {**message, "content": new_blocks}
    return result


def _elided_read_ids(before: list[dict], after: list[dict]) -> set[str]:
    """tool_use ids whose tool_result content was elided by compaction.

    Compares the pre/post message lists block-by-block: a tool_result whose
    inner content string changed was stubbed to fit budget, so any read-dedup
    entry pointing at it must be dropped -- otherwise a repeat read is told
    "[already read above]" while the bytes it named are gone.
    """
    ids: set[str] = set()
    for old, new in zip(before, after):
        old_blocks = old.get("content")
        new_blocks = new.get("content")
        if not isinstance(old_blocks, list) or not isinstance(new_blocks, list):
            continue
        for ob, nb in zip(old_blocks, new_blocks):
            if (
                isinstance(ob, dict)
                and isinstance(nb, dict)
                and ob.get("type") == "tool_result"
                and ob.get("content") != nb.get("content")
            ):
                tid = nb.get("tool_use_id")
                if tid:
                    ids.add(tid)
    return ids


@register_provider("claude")
class ClaudeProvider(BaseProvider):
    """Anthropic Claude API provider.

    Supports both standard API keys and OAuth tokens from Claude Code CLI.
    OAuth tokens (``sk-ant-oat01`` prefix) are proxied through `claude -p`.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.anthropic.com/v1"
        self.client = httpx.Client(timeout=60.0)
        self._use_oauth_cli = config.api_key.startswith("sk-ant-oat01")
        self._claude_bin = shutil.which("claude")
        self._use_cli_proxy = self._use_oauth_cli and bool(self._claude_bin)
        default_activity = "1" if self._use_cli_proxy else "0"
        self._emit_activity = (
            os.getenv("CASCADE_CLAUDE_ACTIVITY", default_activity).lower()
            not in ("0", "false", "no", "off")
        )

    def get_fallback_model(self) -> Optional[str]:
        """Fall back from Claude Opus to Sonnet on rate limits."""
        if "opus" in self.config.model:
            # Newer Opus (4.7/4.8) has no matching Sonnet version -- use the current Sonnet.
            return "claude-sonnet-5"
        return None

    def _headers(self) -> dict:
        return {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _stream_via_cli(
        self,
        messages: list[Message],
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream assistant text by proxying through ``claude -p``."""
        if not self._claude_bin:
            yield "Error: claude CLI not found in PATH for OAuth mode."
            return

        prompt = self._condense_for_cli(messages)
        workdir = self.get_working_directory()
        # LIMITATION: a CLI-proxy provider runs its OWN tools inside the
        # subprocess, which cascade's permission engine cannot see or gate.
        # The sacred-path / dangerous-shell floors therefore do NOT apply to
        # proxy tool calls -- only the coarse posture->CLI-mode mapping
        # below does. The structural floors are enforced on the direct-API
        # providers (openrouter/openai/local/...), Eve's post-08/05 daily
        # drivers. Non-interactive `claude -p` cannot prompt, so posture
        # maps onto its permission modes.
        posture = getattr(
            getattr(self, "permission_engine", None), "posture", "auto",
        )
        # -p is non-interactive: it cannot prompt for approval, so "safe"
        # (mutations must be asked) maps to plan, not acceptEdits -- the
        # latter would silently auto-approve every edit, inverting safe.
        proxy_mode = {
            "auto": "bypassPermissions",
            "safe": "plan",
            "readonly": "plan",
        }.get(posture, "bypassPermissions")
        cmd = [
            self._claude_bin, "-p", prompt,
            "--output-format", "stream-json",
            "--include-partial-messages", "--verbose",
            "--add-dir", workdir,
            "--permission-mode", proxy_mode,
        ]
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        if system:
            cmd.extend(["--system-prompt", system])

        handler = _AccumulatingClaudeEventHandler()
        cfg = CLIProxyConfig(
            binary=self._claude_bin,
            cli_name="claude",
            cmd_args=cmd,
            cwd=workdir,
        )
        cancel_token = self.cancellation_token()
        if cancel_token is None:
            yield from stream_cli_proxy(cfg, handler, self._emit_activity)
        else:
            yield from stream_cli_proxy(
                cfg, handler, self._emit_activity, cancel_token,
            )
        handler.bank_final_turn()
        if handler.total_usage is not None:
            # Accumulated spend across every internal turn; the base handler's
            # last_usage is the final turn only -- the context-occupancy anchor.
            # NEVER fall back to total_usage for the anchor: it is the SUM over
            # all rounds (each re-sends the growing context), so using it as the
            # last-round size inflates the anchor ~Nx-rounds -- which showed as a
            # 7.4M-token "ctx 999%" and made should_compact strip real context.
            # If the final-turn usage is unknown, None -> honest "ctx ?".
            self._last_usage = handler.total_usage
            self._last_round_usage = handler.last_usage
        elif handler.last_usage:
            self._last_usage = handler.last_usage
            self._last_round_usage = handler.last_usage

    def ask(self, messages: list[Message], system: Optional[str] = None) -> str:
        """Get a complete response from Claude."""
        return "".join(self.stream(messages, system))

    def stream(self, messages: list[Message], system: Optional[str] = None) -> Iterator[str]:
        """Stream tokens from Claude."""
        self._last_usage = None
        self._last_round_usage = None
        self.reset_activity_state()
        if self._use_cli_proxy:
            yield from self._filter_activity(self._stream_via_cli(messages, system))
            return
        if self._use_oauth_cli and not self._claude_bin:
            raise RuntimeError(
                "Claude OAuth token detected, but claude CLI is not in PATH."
            )

        try:
            url = f"{self.base_url}/messages"
            api_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in messages
            ]
            payload = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens or 2048,
                "stream": True,
                "messages": api_messages,
            }
            if _accepts_sampling(self.config.model):
                payload["temperature"] = self.config.temperature
            if system:
                # Cache the byte-stable system prompt so a long no-tool chat
                # re-reads it at cache rates (mirrors ask_with_tools). Anthropic
                # accepts system as a string or a content-block list.
                payload["system"] = [
                    {"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}},
                ]

            with self.client.stream(
                "POST", url, json=payload, headers=self._headers()
            ) as response, self.cancellation_callback(
                getattr(response, "close", lambda: None)
            ):
                response.raise_for_status()
                for line in response.iter_lines():
                    self.raise_if_cancelled()
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                if "delta" in data and "text" in data["delta"]:
                                    yield data["delta"]["text"]
                            elif data.get("type") == "message_delta":
                                usage = data.get("usage", {})
                                out_tokens = usage.get("output_tokens", 0)
                                if out_tokens:
                                    prev = self._last_usage or Usage()
                                    self._last_usage = replace(prev, output=out_tokens)
                                    self._last_round_usage = self._last_usage
                            elif data.get("type") == "message_start":
                                usage = data.get("message", {}).get("usage", {})
                                self._last_usage = Usage.from_anthropic(usage)
                                self._last_round_usage = self._last_usage
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(str(exc)) from exc

    def ask_with_tools(
        self,
        messages: list[Message],
        tools: dict[str, "ToolDef"],
        system: Optional[str] = None,
        max_rounds: int = 5,
        on_tool_event: ToolEventCallback = None,
        on_pending_message=None,  # accepted for interface parity; this loop
        # uses dispatch-on-completion (a mid-loop user block would need to ride
        # the tool_result user turn to keep tool_use/tool_result paired).
    ) -> tuple[str, list[dict]]:
        """Claude-native tool calling using tools array + tool_use/tool_result."""
        if self._use_cli_proxy:
            return self.ask(messages, system), []
        self._last_usage = None
        self._last_round_usage = None

        from ..tools.executor import ToolExecutor
        from ..tools.permissions import PermissionAbort

        executor = ToolExecutor(
            tools,
            hook_runner=self.hook_runner,
            permissions=self.permission_engine,
        )
        tool_defs = [
            {
                "name": td.name,
                "description": td.description,
                "input_schema": td.parameters,
            }
            for td in tools.values()
        ]
        # Prompt caching (direct API only -- the CLI proxy manages its own).
        # Render order is tools -> system -> messages, so a breakpoint on the
        # last tool and one on the system block cache the whole stable prefix;
        # only the growing message tail pays full price each round.
        if tool_defs:
            tool_defs[-1] = {**tool_defs[-1], "cache_control": {"type": "ephemeral"}}
        system_blocks = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if system
            else None
        )

        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]
        tool_log = []
        # Bound accumulated tool results and skip re-reading a known path, as the
        # shared OpenAI loop does -- otherwise every file read is re-sent in full
        # on every round.
        budget = int(
            window_for("claude", self.config.model, self.config.context_window)
            * _CONTEXT_BUDGET_FRACTION
        )
        # dedup_key -> the tool_use id whose tool_result holds that read. When
        # budget compaction later ELIDES that result, the key is dropped so a
        # repeat read re-fetches the file instead of "[already read above]"
        # pointing at content that is gone. Mirrors the shared OpenAI loop.
        seen_reads: dict[tuple[str, str], str] = {}

        text_parts = []
        for round_num in range(max_rounds):
            self.raise_if_cancelled()
            _before = api_messages
            api_messages = _compact_anthropic_tool_results(
                api_messages, budget, _KEEP_RECENT_MESSAGES,
            )
            # Drop dedup keys whose read result was just elided.
            if seen_reads and _before is not api_messages:
                elided_ids = _elided_read_ids(_before, api_messages)
                if elided_ids:
                    seen_reads = {
                        key: tid
                        for key, tid in seen_reads.items()
                        if tid not in elided_ids
                    }
            payload = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens or 2048,
                "messages": api_messages,
                "tools": tool_defs,
            }
            if _accepts_sampling(self.config.model):
                payload["temperature"] = self.config.temperature
            if system_blocks:
                payload["system"] = system_blocks

            url = f"{self.base_url}/messages"
            try:
                response = self.client.post(url, json=payload, headers=self._headers())
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(str(exc)) from exc
            except httpx.RequestError as exc:
                raise RuntimeError(str(exc)) from exc
            self.raise_if_cancelled()

            # Capture token usage: accumulate for spend, keep the final
            # round separately as the context anchor.
            round_usage = Usage.from_anthropic(data.get("usage", {}))
            if round_usage.total:
                prev = self._last_usage or Usage()
                self._last_usage = prev.add(round_usage)
                self._last_round_usage = round_usage

            # Check stop reason
            stop_reason = data.get("stop_reason", "end_turn")

            # Extract text and tool_use blocks
            text_parts = []
            tool_uses = []
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_uses.append(block)

            # Execute tool_use blocks whenever present, regardless of stop_reason:
            # Anthropic-compatible cheap endpoints (z.ai/GLM, Moonshot/Kimi) return
            # complete tool_use with stop_reason "end_turn"/"max_tokens", and the
            # exact-match gate dropped them -> the edits never ran ("no changes, no
            # error"). Only a genuinely tool-less round ends the turn.
            if not tool_uses:
                return "".join(text_parts), tool_log

            # Append the assistant message with all content blocks
            api_messages.append({"role": "assistant", "content": data["content"]})

            # Execute each tool call and build tool_result messages
            tool_results = []
            for tool_use in tool_uses:
                self.raise_if_cancelled()
                tool_name = tool_use["name"]
                tool_input = tool_use.get("input", {})
                tool_id = tool_use["id"]

                if on_tool_event:
                    on_tool_event(ToolEvent(
                        kind="tool_start",
                        tool_name=tool_name,
                        round_num=round_num,
                        max_rounds=max_rounds,
                        tool_input=tool_input,
                    ))

                dedup_key = _read_dedup_key(tool_name, tool_input)
                if dedup_key is not None and dedup_key in seen_reads:
                    result = f"[already read above: {dedup_key[1]}]"
                else:
                    try:
                        result = executor.execute(tool_name, tool_input)
                    except PermissionAbort as exc:
                        return ("".join(text_parts) + f"\n\n[stopped: {exc}]").strip(), tool_log
                    if dedup_key is not None:
                        seen_reads[dedup_key] = tool_id
                    else:
                        # A successful edit makes any cached read of that path
                        # stale: drop its dedup keys so the next read re-fetches.
                        edited = _invalidates_read(tool_name, tool_input)
                        if edited is not None and seen_reads:
                            seen_reads = {
                                key: tid
                                for key, tid in seen_reads.items()
                                if key[1] != edited
                            }
                self.raise_if_cancelled()
                tool_log.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "output": result,
                })

                if on_tool_event:
                    on_tool_event(ToolEvent(
                        kind="tool_done",
                        tool_name=tool_name,
                        round_num=round_num,
                        max_rounds=max_rounds,
                        tool_input=tool_input,
                        tool_output=result,
                    ))

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result,
                })

            api_messages.append({"role": "user", "content": tool_results})

        # Exhausted rounds, return whatever text we have
        return "".join(text_parts) if text_parts else "", tool_log

    def compare(self, prompt: str, system: Optional[str] = None) -> dict:
        """Generate comparison data."""
        response = self.ask_single(prompt, system)
        return {
            "provider": self.name,
            "model": self.config.model,
            "response": response,
            "length": len(response),
        }

    def __del__(self):
        """Cleanup HTTP client."""
        try:
            self.client.close()
        except Exception:
            pass
