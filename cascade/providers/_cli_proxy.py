"""Shared CLI proxy infrastructure for subprocess-based provider streaming.

Extracts the common subprocess spawning, JSON event parsing, activity
emission, and error handling logic that was duplicated across the Gemini,
Claude, and OpenAI providers.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Iterator, Optional


ACTIVITY_PREFIX = "[[cascade_activity]] "


@dataclass(frozen=True)
class CLIProxyConfig:
    """Configuration for a CLI proxy subprocess."""

    binary: str
    cli_name: str
    cmd_args: list[str]
    env_overrides: dict[str, str] = field(
        default_factory=lambda: {"NO_BROWSER": "true"},
    )


# ---------------------------------------------------------------------------
# Event handlers — one per CLI provider
# ---------------------------------------------------------------------------


class CLIEventHandler:
    """Base class for provider-specific CLI JSON event handlers."""

    def __init__(self) -> None:
        self.last_usage: Optional[tuple[int, int]] = None
        self.error_lines: list[str] = []
        self.saw_text: bool = False

    def on_json_event(self, event: dict) -> Iterator[tuple[str, str]]:
        """Process a parsed JSON event line.

        Yields ``("text", value)`` for assistant content or
        ``("activity", value)`` for status messages.
        """
        return iter(())

    def on_non_json_line(self, line: str) -> Optional[str]:
        """Process a non-JSON stdout line.

        Return an activity string, or *None* to suppress it.
        """
        return None


class GeminiEventHandler(CLIEventHandler):
    """Handles Gemini CLI ``--output-format stream-json`` events."""

    def on_json_event(self, event: dict) -> Iterator[tuple[str, str]]:
        ev_type = event.get("type")

        if ev_type == "init":
            yield ("activity", f"model: {event.get('model', '')}")

        elif ev_type == "tool_use":
            tool_name = event.get("tool_name", "tool")
            params = event.get("parameters", {})
            try:
                params_text = json.dumps(params, ensure_ascii=True)
            except Exception:
                params_text = str(params)
            if len(params_text) > 140:
                params_text = params_text[:137] + "..."
            yield ("activity", f"tool: {tool_name} {params_text}")

        elif ev_type == "tool_result":
            tool_id = event.get("tool_id", "")
            status_text = event.get("status", "unknown")
            yield ("activity", f"tool result: {tool_id} ({status_text})")

        elif ev_type == "message" and event.get("role") == "assistant":
            text = event.get("content")
            if isinstance(text, str) and text:
                yield ("text", text)

        elif ev_type == "result":
            stats = event.get("stats", {})
            if isinstance(stats, dict):
                in_t = stats.get("input_tokens", 0)
                out_t = stats.get("output_tokens", 0)
                if isinstance(in_t, int) and isinstance(out_t, int):
                    self.last_usage = (in_t, out_t)
                duration = stats.get("duration_ms")
                if isinstance(duration, int):
                    yield ("activity", f"done in {duration}ms")

    def on_non_json_line(self, line: str) -> Optional[str]:
        if line == "Loaded cached credentials.":
            return None
        self.error_lines.append(line)
        return line


class ClaudeEventHandler(CLIEventHandler):
    """Handles Claude CLI ``--output-format stream-json`` events."""

    def __init__(self) -> None:
        super().__init__()
        self.saw_delta: bool = False
        self._thinking: bool = False
        self._current_tool: str = ""

    def on_json_event(self, event: dict) -> Iterator[tuple[str, str]]:
        ev_type = event.get("type")

        if ev_type == "system":
            if event.get("subtype") == "init":
                model = event.get("model")
                if isinstance(model, str) and model:
                    yield ("activity", f"model: {model}")
                tools = event.get("tools")
                if isinstance(tools, list) and tools:
                    names = [
                        t.get("name", "?") if isinstance(t, dict) else str(t)
                        for t in tools[:5]
                    ]
                    yield ("activity", f"tools: {', '.join(names)}")

        elif ev_type == "stream_event":
            inner = event.get("event", {})
            if isinstance(inner, dict):
                yield from self._handle_stream_event(inner)

        elif ev_type == "assistant":
            if not self.saw_delta:
                message = event.get("message", {})
                if isinstance(message, dict):
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    yield ("text", text)
                            elif isinstance(block, str) and block:
                                yield ("text", block)
                    usage = message.get("usage", {})
                    if isinstance(usage, dict):
                        in_t = usage.get("input_tokens")
                        out_t = usage.get("output_tokens")
                        if isinstance(in_t, int) and isinstance(out_t, int):
                            self.last_usage = (in_t, out_t)

        elif ev_type == "result":
            usage = event.get("usage", {})
            if isinstance(usage, dict):
                in_t = usage.get("input_tokens")
                out_t = usage.get("output_tokens")
                if isinstance(in_t, int) and isinstance(out_t, int):
                    self.last_usage = (in_t, out_t)
            duration = event.get("duration_ms")
            if isinstance(duration, int):
                yield ("activity", f"done in {duration}ms")
            if event.get("is_error"):
                msg = event.get("result")
                if isinstance(msg, str) and msg:
                    self.error_lines.append(msg)

        elif ev_type == "rate_limit_event":
            info = event.get("rate_limit_info", {})
            if isinstance(info, dict):
                resets = info.get("resetsAt")
                if isinstance(resets, int):
                    yield ("activity", f"five-hour window resets at {resets}")

    def _handle_stream_event(self, inner: dict) -> Iterator[tuple[str, str]]:
        inner_type = inner.get("type")

        if inner_type == "content_block_start":
            block = inner.get("content_block", {})
            block_type = block.get("type")
            if block_type == "tool_use":
                self._current_tool = block.get("name", "tool")
                yield ("activity", f"calling tool: {self._current_tool}")
            elif block_type == "thinking":
                self._thinking = True
                yield ("activity", "thinking...")

        elif inner_type == "content_block_delta":
            delta = inner.get("delta", {})
            delta_type = delta.get("type", "")

            if delta_type == "thinking_delta":
                # Show a truncated snippet of thinking as activity
                thought = delta.get("thinking", "")
                if isinstance(thought, str) and thought:
                    snippet = thought.strip().replace("\n", " ")
                    if len(snippet) > 80:
                        snippet = snippet[:77] + "..."
                    if snippet:
                        yield ("activity", f"thinking: {snippet}")
            elif delta_type == "input_json_delta":
                # Tool input streaming — show what tool is receiving
                partial = delta.get("partial_json", "")
                if self._current_tool and isinstance(partial, str) and len(partial) > 2:
                    snippet = partial.strip()
                    if len(snippet) > 80:
                        snippet = snippet[:77] + "..."
                    yield ("activity", f"{self._current_tool}: {snippet}")
            else:
                # text_delta or untyped delta — yield as text
                text = delta.get("text")
                if isinstance(text, str) and text:
                    self.saw_delta = True
                    yield ("text", text)

        elif inner_type == "content_block_stop":
            if self._thinking:
                self._thinking = False
                yield ("activity", "thinking complete")
            if self._current_tool:
                yield ("activity", f"{self._current_tool} call complete")
                self._current_tool = ""

        elif inner_type == "message_start":
            usage = inner.get("message", {}).get("usage", {})
            in_t = usage.get("input_tokens", 0)
            if isinstance(in_t, int):
                self.last_usage = (in_t, 0)

        elif inner_type == "message_delta":
            out_t = inner.get("usage", {}).get("output_tokens")
            if isinstance(out_t, int):
                prev = self.last_usage or (0, 0)
                self.last_usage = (prev[0], out_t)

    def on_non_json_line(self, line: str) -> Optional[str]:
        self.error_lines.append(line)
        return line


class CodexEventHandler(CLIEventHandler):
    """Handles Codex CLI ``--json`` events."""

    @staticmethod
    def _extract_text(item: dict) -> str:
        """Extract assistant text from a Codex item payload."""
        text = item.get("text")
        if isinstance(text, str) and text:
            return text
        content = item.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "output_text":
                    chunk = block.get("text")
                    if isinstance(chunk, str):
                        parts.append(chunk)
            if parts:
                return "".join(parts)
        return ""

    def on_json_event(self, event: dict) -> Iterator[tuple[str, str]]:
        ev_type = event.get("type")

        if ev_type == "thread.started":
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                yield ("activity", f"thread: {thread_id[:12]}...")

        elif ev_type == "turn.started":
            yield ("activity", "turn started")

        elif ev_type == "error":
            msg = event.get("message")
            if isinstance(msg, str) and msg:
                self.error_lines.append(msg)
                yield ("activity", msg)

        elif ev_type == "item.completed":
            yield from self._handle_item(event.get("item", {}))

        elif ev_type == "turn.completed":
            usage = event.get("usage", {})
            in_t = usage.get("input_tokens")
            out_t = usage.get("output_tokens")
            if isinstance(in_t, int) and isinstance(out_t, int):
                self.last_usage = (in_t, out_t)
            yield ("activity", "turn completed")

    def _handle_item(self, item: dict) -> Iterator[tuple[str, str]]:
        item_type = item.get("type")
        if item_type == "agent_message":
            text = self._extract_text(item)
            if text:
                yield ("text", text)
        elif item_type == "reasoning":
            reason = item.get("text")
            if isinstance(reason, str) and reason:
                yield ("activity", reason)
        elif item_type == "error":
            msg = item.get("message")
            if isinstance(msg, str) and msg:
                self.error_lines.append(msg)
                yield ("activity", msg)

    def on_non_json_line(self, line: str) -> Optional[str]:
        self.error_lines.append(line)
        return line


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _terminate_with_timeout(
    proc: subprocess.Popen,
    grace_seconds: float = 5.0,
) -> None:
    """Send SIGTERM, wait *grace_seconds*, then SIGKILL if still alive."""
    proc.terminate()
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def stream_cli_proxy(
    config: CLIProxyConfig,
    handler: CLIEventHandler,
    emit_activity: bool = True,
) -> Iterator[str]:
    """Spawn a CLI subprocess and stream parsed output.

    Yields plain text chunks (assistant content) and activity-prefixed
    status messages.  The caller is expected to pass the output through
    ``BaseProvider._filter_activity`` to separate the two.

    The subprocess runs until it exits naturally.  Use Ctrl+C to cancel.
    """
    env = os.environ.copy()
    for key, value in config.env_overrides.items():
        env.setdefault(key, value)

    try:
        proc = subprocess.Popen(
            config.cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:
        yield f"Error: {exc}"
        return

    if emit_activity:
        yield f"{ACTIVITY_PREFIX}starting {config.cli_name} cli in {os.getcwd()}"

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            activity = handler.on_non_json_line(line)
            if activity is not None and emit_activity:
                yield f"{ACTIVITY_PREFIX}{activity}"
            continue

        for kind, value in handler.on_json_event(event):
            if kind == "text":
                handler.saw_text = True
                yield value
            elif kind == "activity" and emit_activity:
                yield f"{ACTIVITY_PREFIX}{value}"

    proc.wait()

    if proc.returncode != 0 and not handler.saw_text:
        msg = (
            handler.error_lines[-1]
            if handler.error_lines
            else f"{config.cli_name} exited with code {proc.returncode}"
        )
        yield f"Error: {msg}"
    elif not handler.saw_text and handler.error_lines:
        yield f"Error: {handler.error_lines[-1]}"
