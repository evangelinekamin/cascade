"""Tests for the OpenAI provider."""

from unittest.mock import patch

from cascade.providers.base import ProviderConfig
from cascade.providers.openai_provider import OpenAIProvider


def test_openai_accepts_provider_config():
    """OpenAIProvider should accept a ProviderConfig."""
    config = ProviderConfig(
        api_key="test-key",
        model="gpt-4o",
    )
    provider = OpenAIProvider(config)
    assert provider.config is config
    assert provider.config.api_key == "test-key"
    assert provider.config.model == "gpt-4o"


def test_openai_abc_compliance():
    """OpenAIProvider should implement all BaseProvider abstract methods."""
    config = ProviderConfig(api_key="test-key", model="test-model")
    provider = OpenAIProvider(config)
    assert hasattr(provider, "ask")
    assert hasattr(provider, "stream")
    assert hasattr(provider, "compare")
    assert callable(provider.ask)
    assert callable(provider.stream)
    assert callable(provider.compare)


def test_openai_default_base_url():
    """Should use OpenAI base URL by default."""
    config = ProviderConfig(api_key="test-key", model="gpt-4o")
    provider = OpenAIProvider(config)
    assert provider.base_url == "https://api.openai.com/v1"


def test_openai_custom_base_url():
    """Should accept custom base URL for Azure/proxies."""
    config = ProviderConfig(
        api_key="test-key",
        model="gpt-4",
        base_url="https://my-azure.openai.azure.com/v1",
    )
    provider = OpenAIProvider(config)
    assert provider.base_url == "https://my-azure.openai.azure.com/v1"


def test_openai_validation():
    """Should validate with valid config."""
    config = ProviderConfig(api_key="sk-test", model="gpt-4o")
    provider = OpenAIProvider(config)
    assert provider.validate() is True


def test_openai_validation_no_key():
    """Should fail validation without API key."""
    config = ProviderConfig(api_key="", model="gpt-4o")
    provider = OpenAIProvider(config)
    assert provider.validate() is False


def test_uses_cli_proxy_for_codex_oauth_when_binary_exists():
    """JWT OAuth token should route through codex CLI when available."""
    with patch("cascade.providers.openai_provider.shutil.which", return_value="/usr/bin/codex"):
        provider = OpenAIProvider(
            ProviderConfig(api_key="eyJ.a.b", model="gpt-5.3-codex")
        )
    assert provider._use_cli_proxy is True


def test_does_not_use_cli_proxy_for_standard_api_key():
    """Regular OpenAI API keys should keep direct API path."""
    with patch("cascade.providers.openai_provider.shutil.which", return_value="/usr/bin/codex"):
        provider = OpenAIProvider(
            ProviderConfig(api_key="sk-test-key", model="gpt-4o")
        )
    assert provider._use_cli_proxy is False


def test_stream_cli_parses_agent_message_and_usage():
    """Codex JSONL output should yield assistant text and capture usage."""

    class _FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = iter(
                [
                    '{"type":"thread.started","thread_id":"t"}\n',
                    '{"type":"turn.started"}\n',
                    '{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}\n',
                    '{"type":"turn.completed","usage":{"input_tokens":9,"output_tokens":2}}\n',
                ]
            )
            self.returncode = 0

        def wait(self):
            return 0

    with patch("cascade.providers.openai_provider.shutil.which", return_value="/usr/bin/codex"):
        with patch.dict("os.environ", {"CASCADE_OPENAI_ACTIVITY": "0"}, clear=False):
            provider = OpenAIProvider(
                ProviderConfig(api_key="eyJ.a.b", model="gpt-5.3-codex")
            )

    with patch("cascade.providers._cli_proxy.subprocess.Popen", _FakePopen):
        chunks = list(provider.stream_single("Reply with OK"))

    assert chunks == ["OK"]
    assert provider.last_usage == (9, 2)


def test_oauth_token_without_codex_binary_returns_clear_error():
    """OAuth token without codex CLI should not fall back to raw API."""
    with patch("cascade.providers.openai_provider.shutil.which", return_value=None):
        provider = OpenAIProvider(
            ProviderConfig(api_key="eyJ.a.b", model="gpt-5.3-codex")
        )

    chunks = list(provider.stream_single("hello"))
    assert chunks == ["Error: Codex OAuth token detected, but codex CLI is not in PATH."]
