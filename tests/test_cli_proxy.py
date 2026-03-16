"""Tests for shared CLI proxy infrastructure."""

from unittest.mock import patch

from cascade.providers._cli_proxy import (
    ACTIVITY_PREFIX,
    CLIEventHandler,
    CLIProxyConfig,
    ClaudeEventHandler,
    CodexEventHandler,
    GeminiEventHandler,
    stream_cli_proxy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakePopen:
    """Minimal subprocess.Popen stand-in driven by a list of stdout lines."""

    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self.returncode = returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        pass


def _make_popen_cls(lines, returncode=0):
    """Return a Popen class whose instances emit *lines* on stdout."""
    class _Cls:
        def __init__(self, *_a, **_kw):
            self.stdout = iter(lines)
            self.returncode = returncode

        def wait(self, timeout=None):
            return self.returncode

        def terminate(self):
            pass

        def kill(self):
            pass

    return _Cls


def _cfg(**overrides):
    defaults = dict(binary="/usr/bin/test", cli_name="test", cmd_args=["/usr/bin/test"])
    defaults.update(overrides)
    return CLIProxyConfig(**defaults)


# ---------------------------------------------------------------------------
# stream_cli_proxy integration tests
# ---------------------------------------------------------------------------


def test_stream_yields_text_from_handler():
    """Happy path: handler emits text, stream_cli_proxy yields it."""
    lines = [
        '{"type":"message","role":"assistant","content":"hello"}\n',
        '{"type":"result","stats":{"input_tokens":5,"output_tokens":1}}\n',
    ]
    handler = GeminiEventHandler()
    popen_cls = _make_popen_cls(lines)

    with patch("cascade.providers._cli_proxy.subprocess.Popen", popen_cls):
        chunks = list(stream_cli_proxy(_cfg(cli_name="gemini"), handler, emit_activity=False))

    assert chunks == ["hello"]
    assert handler.last_usage == (5, 1)


def test_handles_non_json_lines():
    """Non-JSON lines route through handler.on_non_json_line."""
    lines = [
        "some status message\n",
        '{"type":"message","role":"assistant","content":"ok"}\n',
    ]
    handler = GeminiEventHandler()
    popen_cls = _make_popen_cls(lines)

    with patch("cascade.providers._cli_proxy.subprocess.Popen", popen_cls):
        chunks = list(stream_cli_proxy(_cfg(), handler, emit_activity=False))

    assert chunks == ["ok"]
    assert handler.error_lines == ["some status message"]


def test_activity_messages_emitted():
    """Activity messages should be prefixed when emit_activity is True."""
    lines = ['{"type":"init","model":"test-model"}\n']
    handler = GeminiEventHandler()
    popen_cls = _make_popen_cls(lines)

    with patch("cascade.providers._cli_proxy.subprocess.Popen", popen_cls):
        chunks = list(stream_cli_proxy(_cfg(cli_name="gemini"), handler, emit_activity=True))

    # Should have the start activity + model activity
    activity_chunks = [c for c in chunks if c.startswith(ACTIVITY_PREFIX)]
    assert len(activity_chunks) == 2
    assert "starting gemini cli" in activity_chunks[0]
    assert "model: test-model" in activity_chunks[1]


def test_usage_captured_in_handler():
    """Handler should capture usage from result events."""
    lines = [
        '{"type":"result","stats":{"input_tokens":100,"output_tokens":50}}\n',
    ]
    handler = GeminiEventHandler()
    popen_cls = _make_popen_cls(lines)

    with patch("cascade.providers._cli_proxy.subprocess.Popen", popen_cls):
        list(stream_cli_proxy(_cfg(), handler, emit_activity=False))

    assert handler.last_usage == (100, 50)


def test_error_on_nonzero_exit():
    """Nonzero exit with no text should yield an error."""
    lines = ["something went wrong\n"]
    handler = GeminiEventHandler()
    popen_cls = _make_popen_cls(lines, returncode=1)

    with patch("cascade.providers._cli_proxy.subprocess.Popen", popen_cls):
        chunks = list(stream_cli_proxy(_cfg(cli_name="test"), handler, emit_activity=False))

    assert len(chunks) == 1
    assert chunks[0] == "Error: something went wrong"


# ---------------------------------------------------------------------------
# GeminiEventHandler unit tests
# ---------------------------------------------------------------------------


def test_gemini_handler_event_types():
    handler = GeminiEventHandler()

    # init event
    events = list(handler.on_json_event({"type": "init", "model": "gemini-3.1-pro"}))
    assert events == [("activity", "model: gemini-3.1-pro")]

    # tool_use event
    events = list(handler.on_json_event({
        "type": "tool_use", "tool_name": "read_file", "parameters": {"path": "/tmp"},
    }))
    assert len(events) == 1
    assert events[0][0] == "activity"
    assert "read_file" in events[0][1]

    # message event
    events = list(handler.on_json_event({
        "type": "message", "role": "assistant", "content": "hi",
    }))
    assert events == [("text", "hi")]

    # result event with usage
    events = list(handler.on_json_event({
        "type": "result", "stats": {"input_tokens": 10, "output_tokens": 5, "duration_ms": 200},
    }))
    assert handler.last_usage == (10, 5)
    assert ("activity", "done in 200ms") in events

    # non-JSON line: skip cached credentials
    assert handler.on_non_json_line("Loaded cached credentials.") is None
    assert handler.on_non_json_line("retry in 2s") == "retry in 2s"


# ---------------------------------------------------------------------------
# ClaudeEventHandler unit tests
# ---------------------------------------------------------------------------


def test_claude_handler_stream_events():
    handler = ClaudeEventHandler()

    # system init
    events = list(handler.on_json_event({
        "type": "system", "subtype": "init", "model": "claude-sonnet-4-6",
    }))
    assert events == [("activity", "model: claude-sonnet-4-6")]

    # content_block_delta
    events = list(handler.on_json_event({
        "type": "stream_event",
        "event": {"type": "content_block_delta", "delta": {"text": "Hi"}},
    }))
    assert events == [("text", "Hi")]
    assert handler.saw_delta is True

    # assistant event should be skipped when saw_delta is True
    events = list(handler.on_json_event({
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "Hi"}], "usage": {"input_tokens": 5, "output_tokens": 2}},
    }))
    assert events == []

    # result with usage
    events = list(handler.on_json_event({
        "type": "result",
        "usage": {"input_tokens": 20, "output_tokens": 10},
        "duration_ms": 500,
    }))
    assert handler.last_usage == (20, 10)
    assert ("activity", "done in 500ms") in events

    # rate_limit_event
    events = list(handler.on_json_event({
        "type": "rate_limit_event",
        "rate_limit_info": {"resetsAt": 1234567890},
    }))
    assert ("activity", "five-hour window resets at 1234567890") in events


def test_claude_handler_thinking_and_tool_activity():
    """Claude handler should surface thinking blocks and tool calls as activity."""
    handler = ClaudeEventHandler()

    # thinking block start
    events = list(handler.on_json_event({
        "type": "stream_event",
        "event": {"type": "content_block_start", "content_block": {"type": "thinking"}},
    }))
    assert ("activity", "thinking...") in events

    # thinking delta
    events = list(handler.on_json_event({
        "type": "stream_event",
        "event": {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "Let me consider"}},
    }))
    assert len(events) == 1
    assert events[0][0] == "activity"
    assert "thinking:" in events[0][1]

    # thinking block stop
    events = list(handler.on_json_event({
        "type": "stream_event",
        "event": {"type": "content_block_stop"},
    }))
    assert ("activity", "thinking complete") in events

    # tool_use block start
    handler2 = ClaudeEventHandler()
    events = list(handler2.on_json_event({
        "type": "stream_event",
        "event": {"type": "content_block_start", "content_block": {"type": "tool_use", "name": "read_file"}},
    }))
    assert ("activity", "calling tool: read_file") in events

    # tool input delta
    events = list(handler2.on_json_event({
        "type": "stream_event",
        "event": {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": '{"path":"/tmp/test.py"}'}},
    }))
    assert len(events) == 1
    assert "read_file" in events[0][1]

    # tool block stop
    events = list(handler2.on_json_event({
        "type": "stream_event",
        "event": {"type": "content_block_stop"},
    }))
    assert ("activity", "read_file call complete") in events


def test_claude_handler_assistant_fallback():
    """Without saw_delta, assistant event should yield text."""
    handler = ClaudeEventHandler()

    events = list(handler.on_json_event({
        "type": "assistant",
        "message": {
            "content": [{"type": "text", "text": "hello world"}],
            "usage": {"input_tokens": 8, "output_tokens": 3},
        },
    }))
    assert ("text", "hello world") in events
    assert handler.last_usage == (8, 3)


# ---------------------------------------------------------------------------
# CodexEventHandler unit tests
# ---------------------------------------------------------------------------


def test_codex_handler_item_completed():
    handler = CodexEventHandler()

    # thread.started
    events = list(handler.on_json_event({
        "type": "thread.started", "thread_id": "abcdef123456789",
    }))
    assert len(events) == 1
    assert events[0][0] == "activity"

    # item.completed with agent_message (text field)
    events = list(handler.on_json_event({
        "type": "item.completed",
        "item": {"type": "agent_message", "text": "Done."},
    }))
    assert events == [("text", "Done.")]

    # item.completed with agent_message (content array)
    events = list(handler.on_json_event({
        "type": "item.completed",
        "item": {
            "type": "agent_message",
            "content": [{"type": "output_text", "text": "part1"}, {"type": "output_text", "text": "part2"}],
        },
    }))
    assert events == [("text", "part1part2")]

    # item.completed with reasoning
    events = list(handler.on_json_event({
        "type": "item.completed",
        "item": {"type": "reasoning", "text": "thinking..."},
    }))
    assert events == [("activity", "thinking...")]

    # turn.completed with usage
    events = list(handler.on_json_event({
        "type": "turn.completed",
        "usage": {"input_tokens": 15, "output_tokens": 7},
    }))
    assert handler.last_usage == (15, 7)
    assert ("activity", "turn completed") in events

    # error event
    events = list(handler.on_json_event({
        "type": "error", "message": "rate limited",
    }))
    assert ("activity", "rate limited") in events
    assert "rate limited" in handler.error_lines
