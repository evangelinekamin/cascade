"""Tests for Claude provider auth/path behavior."""

from unittest.mock import patch

from cascade.providers.base import ProviderConfig
from cascade.providers.claude import ClaudeProvider


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
    assert provider.last_usage == (11, 3)


def test_oauth_token_without_claude_binary_returns_clear_error():
    """OAuth token without claude CLI should not fall back to raw API."""
    with patch("cascade.providers.claude.shutil.which", return_value=None):
        provider = ClaudeProvider(
            ProviderConfig(api_key="sk-ant-oat01-test-token", model="claude-opus-4-6")
        )

    chunks = list(provider.stream_single("hello"))
    assert chunks == ["Error: Claude OAuth token detected, but claude CLI is not in PATH."]
