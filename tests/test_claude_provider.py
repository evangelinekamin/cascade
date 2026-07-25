"""Tests for Claude provider auth/path behavior."""

from unittest.mock import patch

import pytest

from cascade.providers.base import ProviderConfig
from cascade.providers.claude import ClaudeProvider, _cache_last_message
from cascade.providers.usage import Usage


def test_cache_last_message_wraps_string_content_with_breakpoint():
    msgs = [{"role": "user", "content": "hello"}]
    out = _cache_last_message(msgs)
    assert out[0]["content"] == [
        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
    ]
    # The caller's list must be untouched -- the tool loop reuses it every round,
    # so a persisted breakpoint would accumulate past Anthropic's limit of four.
    assert msgs[0]["content"] == "hello"


def test_cache_last_message_marks_only_the_tail_block_of_list_content():
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "x"},
            {"type": "tool_result", "tool_use_id": "b", "content": "y"},
        ]},
    ]
    out = _cache_last_message(msgs)
    last_blocks = out[-1]["content"]
    assert last_blocks[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in last_blocks[0]
    assert out[0] is msgs[0]  # earlier messages shared, not copied
    assert "cache_control" not in msgs[-1]["content"][-1]  # original untouched


def test_cache_last_message_noops_on_empty():
    assert _cache_last_message([]) == []
    empty = [{"role": "user", "content": ""}]
    assert _cache_last_message(empty) is empty


class _OkAnthropicStream:
    """A minimal successful /messages stream context manager."""

    class _Response:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}'
            yield 'data: {"type":"content_block_delta","delta":{"text":"OK"}}'

    def __enter__(self):
        return self._Response()

    def __exit__(self, exc_type, exc, tb):
        return False


def test_direct_api_stream_rolls_cache_breakpoint_onto_last_message():
    """The direct-API streaming payload marks the newest message for caching."""
    with patch("cascade.providers.claude.shutil.which", return_value="/usr/bin/claude"):
        provider = ClaudeProvider(
            ProviderConfig(api_key="sk-ant-api03-test-key", model="claude-opus-4-8")
        )
    assert provider._use_cli_proxy is False

    with patch.object(provider.client, "stream", return_value=_OkAnthropicStream()) as mock_stream:
        list(provider.stream_single("plan the thing"))

    payload = mock_stream.call_args.kwargs["json"]
    assert payload["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_uses_cli_proxy_for_oauth_token_when_claude_binary_exists():
    """OAuth token should route through claude CLI when available."""
    with patch("cascade.providers.claude.shutil.which", return_value="/usr/bin/claude"):
        provider = ClaudeProvider(
            ProviderConfig(api_key="sk-ant-oat01-test-token", model="claude-opus-4-6")
        )
    assert provider._use_cli_proxy is True


def test_does_not_use_cli_proxy_for_regular_api_key():
    """Regular API key should keep direct API path."""
    with patch("cascade.providers.claude.shutil.which", return_value="/usr/bin/claude"):
        provider = ClaudeProvider(
            ProviderConfig(api_key="sk-ant-api03-test-key", model="claude-opus-4-6")
        )
    assert provider._use_cli_proxy is False


def test_stream_cli_parses_deltas_and_usage():
    """Claude stream-json output should yield deltas and capture usage."""

    class _FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = iter(
                [
                    '{"type":"system","subtype":"init","model":"claude-sonnet-4-6"}\n',
                    '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"text":"Hel"}}}\n',
                    '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"text":"lo"}}}\n',
                    '{"type":"result","subtype":"success","usage":{"input_tokens":11,"output_tokens":3}}\n',
                ]
            )
            self.returncode = 0

        def wait(self):
            return 0

    with patch("cascade.providers.claude.shutil.which", return_value="/usr/bin/claude"):
        with patch.dict("os.environ", {"CASCADE_CLAUDE_ACTIVITY": "0"}, clear=False):
            provider = ClaudeProvider(
                ProviderConfig(api_key="sk-ant-oat01-test-token", model="claude-opus-4-6")
            )

    with patch("cascade.providers._cli_proxy.subprocess.Popen", _FakePopen):
        chunks = list(provider.stream_single("Say hello"))

    assert chunks == ["Hel", "lo"]
    assert provider.last_usage == Usage(input=11, output=3)


def test_oauth_token_without_claude_binary_raises_clear_error():
    """OAuth token without claude CLI raises instead of yielding error text.

    Yielded error strings get recorded as successful assistant messages;
    a raise surfaces through the FAILED-run path.
    """
    with patch("cascade.providers.claude.shutil.which", return_value=None):
        provider = ClaudeProvider(
            ProviderConfig(api_key="sk-ant-oat01-test-token", model="claude-opus-4-6")
        )

    with pytest.raises(RuntimeError, match="claude CLI is not in PATH"):
        list(provider.stream_single("hello"))


def test_stream_cli_raises_on_authentication_failure_payload():
    """Expired Claude OAuth output should raise instead of streaming fake assistant text."""

    class _FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = iter(
                [
                    '{"type":"system","subtype":"init","model":"claude-opus-4-6"}\n',
                    (
                        '{"type":"assistant","message":{"content":'
                        '[{"type":"text","text":"Failed to authenticate. API Error: 401 '
                        '{\\"type\\":\\"error\\",\\"error\\":{\\"type\\":\\"authentication_error\\",'
                        '\\"message\\":\\"OAuth token has expired.\\"}}"}],'
                        '"usage":{"input_tokens":0,"output_tokens":0}},'
                        '"error":"authentication_failed"}\n'
                    ),
                    (
                        '{"type":"result","subtype":"success","is_error":true,'
                        '"result":"Failed to authenticate. API Error: 401",'
                        '"usage":{"input_tokens":0,"output_tokens":0}}\n'
                    ),
                ]
            )
            self.returncode = 0

        def wait(self):
            return 0

    with patch("cascade.providers.claude.shutil.which", return_value="/usr/bin/claude"):
        with patch.dict("os.environ", {"CASCADE_CLAUDE_ACTIVITY": "0"}, clear=False):
            provider = ClaudeProvider(
                ProviderConfig(api_key="sk-ant-oat01-test-token", model="claude-opus-4-6")
            )

    with patch("cascade.providers._cli_proxy.subprocess.Popen", _FakePopen):
        with pytest.raises(RuntimeError, match="OAuth token has expired"):
            list(provider.stream_single("Say hello"))
