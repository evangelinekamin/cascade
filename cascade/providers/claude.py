"""Anthropic Claude provider implementation."""

from typing import Optional, Iterator, TYPE_CHECKING
import json
import os
import shutil
import httpx
from dataclasses import replace
from .base import BaseProvider, ProviderConfig, Message, ToolEvent, ToolEventCallback
from ._cli_proxy import CLIProxyConfig, ClaudeEventHandler, stream_cli_proxy
from .registry import register_provider
from .usage import Usage

if TYPE_CHECKING:
    from ..tools.schema import ToolDef


# Opus 4.7+, Sonnet 5, and the Fable/Mythos family reject sampling params
# (temperature / top_p / top_k) with a 400. Older models still accept them.
_NO_SAMPLING_TAGS = ("opus-4-7", "opus-4-8", "sonnet-5", "fable-5", "mythos")


def _accepts_sampling(model: str) -> bool:
    return not any(tag in model for tag in _NO_SAMPLING_TAGS)


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
        # Non-interactive `claude -p` cannot prompt for approvals, so the
        # posture maps onto its permission modes: auto keeps bypass (the
        # subprocess runs its own tools; cascade-side gates cannot reach
        # them), safe limits it to edits, readonly plans only.
        posture = getattr(
            getattr(self, "permission_engine", None), "posture", "auto",
        )
        proxy_mode = {
            "auto": "bypassPermissions",
            "safe": "acceptEdits",
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

        handler = ClaudeEventHandler()
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
        if handler.last_usage:
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
                payload["system"] = system

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
    ) -> tuple[str, list[dict]]:
        """Claude-native tool calling using tools array + tool_use/tool_result."""
        if self._use_cli_proxy:
            return self.ask(messages, system), []
        self._last_usage = None
        self._last_round_usage = None

        from ..tools.executor import ToolExecutor

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

        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]
        tool_log = []

        text_parts = []
        for round_num in range(max_rounds):
            self.raise_if_cancelled()
            payload = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens or 2048,
                "messages": api_messages,
                "tools": tool_defs,
            }
            if _accepts_sampling(self.config.model):
                payload["temperature"] = self.config.temperature
            if system:
                payload["system"] = system

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

            if not tool_uses or stop_reason != "tool_use":
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

                result = executor.execute(tool_name, tool_input)
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
