"""Tests for the OpenRouter provider."""

from unittest.mock import patch

import httpx
import pytest

from cascade.providers.base import ProviderConfig
from cascade.providers.openrouter import OpenRouterProvider


def test_openrouter_accepts_provider_config():
    """OpenRouterProvider should accept a ProviderConfig (not a dict)."""
    config = ProviderConfig(
        api_key="test-key",
        model="qwen/qwen3.5-9b",
    )
    provider = OpenRouterProvider(config)
    assert provider.config is config
    assert provider.config.api_key == "test-key"
    assert provider.config.model == "qwen/qwen3.5-9b"


def test_openrouter_abc_compliance():
    """OpenRouterProvider should implement all BaseProvider abstract methods."""
    config = ProviderConfig(api_key="test-key", model="test-model")
    provider = OpenRouterProvider(config)
    assert hasattr(provider, "ask")
    assert hasattr(provider, "stream")
    assert hasattr(provider, "compare")
    assert callable(provider.ask)
    assert callable(provider.stream)
    assert callable(provider.compare)


def test_openrouter_default_base_url():
    """Should use OpenRouter base URL by default."""
    config = ProviderConfig(api_key="test-key", model="test-model")
    provider = OpenRouterProvider(config)
    assert provider.base_url == "https://openrouter.ai/api/v1"


def test_openrouter_custom_base_url():
    """Should accept a custom base URL."""
    config = ProviderConfig(
        api_key="test-key",
        model="test-model",
        base_url="https://custom.api/v1",
    )
    provider = OpenRouterProvider(config)
    assert provider.base_url == "https://custom.api/v1"


def test_openrouter_validation():
    """Should validate with valid config."""
    config = ProviderConfig(api_key="test-key", model="test-model")
    provider = OpenRouterProvider(config)
    assert provider.validate() is True


def test_openrouter_validation_no_key():
    """Should fail validation without API key."""
    config = ProviderConfig(api_key="", model="test-model")
    provider = OpenRouterProvider(config)
    assert provider.validate() is False


def test_openrouter_headers_include_app_attribution():
    """OpenRouter requests should identify the Cascade app correctly."""
    config = ProviderConfig(api_key="test-key", model="test-model")
    provider = OpenRouterProvider(config)

    headers = provider._headers()

    assert headers["HTTP-Referer"] == "https://github.com/Evangeline-Development-Company/cascade"
    assert headers["X-OpenRouter-Title"] == "Cascade"


def test_openrouter_stream_raises_on_http_status_error():
    """HTTP errors should raise, not be returned as assistant text."""
    config = ProviderConfig(api_key="test-key", model="test-model")
    provider = OpenRouterProvider(config)

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(503, request=request)

    class _StreamContext:
        def __enter__(self):
            return response

        def __exit__(self, exc_type, exc, tb):
            return False

    with patch.object(provider.client, "stream", return_value=_StreamContext()):
        with pytest.raises(RuntimeError, match="503"):
            list(provider.stream_single("Reply with exactly OK."))


def test_openrouter_stream_falls_back_on_retryable_status():
    """Retryable provider-routing failures should retry once with the fallback model."""
    config = ProviderConfig(
        api_key="test-key",
        model="qwen/qwen3.5-9b",
        fallback_model="minimax/minimax-m2.5",
    )
    provider = OpenRouterProvider(config)

    failing_request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    failing_response = httpx.Response(503, request=failing_request)

    class _FailingContext:
        def __enter__(self):
            return failing_response

        def __exit__(self, exc_type, exc, tb):
            return False

    class _SuccessResponse:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"OK"}}]}'
            yield 'data: {"usage":{"prompt_tokens":7,"completion_tokens":2},"choices":[]}'
            yield "data: [DONE]"

    class _SuccessContext:
        def __enter__(self):
            return _SuccessResponse()

        def __exit__(self, exc_type, exc, tb):
            return False

    with patch.object(provider.client, "stream", side_effect=[_FailingContext(), _SuccessContext()]) as mock_stream:
        chunks = list(provider.stream_single("Reply with exactly OK."))

    assert chunks == ["OK"]
    assert provider.last_usage == (7, 2)
    first_call = mock_stream.call_args_list[0].kwargs["json"]
    second_call = mock_stream.call_args_list[1].kwargs["json"]
    assert first_call["model"] == "qwen/qwen3.5-9b"
    assert second_call["model"] == "minimax/minimax-m2.5"


def _ok_stream_context():
    """A one-line successful chat-completions stream context manager."""

    class _SuccessResponse:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"OK"}}]}'
            yield "data: [DONE]"

    class _SuccessContext:
        def __enter__(self):
            return _SuccessResponse()

        def __exit__(self, exc_type, exc, tb):
            return False

    return _SuccessContext()


def test_provider_config_defaults_provider_preferences_to_none():
    """A ProviderConfig without explicit preferences leaves them unset."""
    config = ProviderConfig(api_key="k", model="m")
    assert config.provider_preferences is None


def test_openrouter_stream_pins_provider_when_preferences_set():
    """The streaming payload carries the configured upstream 'provider' order."""
    prefs = {"order": ["Baidu", "Fireworks", "Alibaba"], "allow_fallbacks": True}
    config = ProviderConfig(
        api_key="test-key",
        model="qwen/qwen3.5-9b",
        provider_preferences=prefs,
    )
    provider = OpenRouterProvider(config)

    with patch.object(provider.client, "stream", return_value=_ok_stream_context()) as mock_stream:
        list(provider.stream_single("Reply with exactly OK."))

    payload = mock_stream.call_args.kwargs["json"]
    assert payload["provider"] == prefs


def test_openrouter_stream_omits_provider_when_preferences_none():
    """With no preferences, the streaming payload has no 'provider' field."""
    config = ProviderConfig(api_key="test-key", model="qwen/qwen3.5-9b")
    provider = OpenRouterProvider(config)

    with patch.object(provider.client, "stream", return_value=_ok_stream_context()) as mock_stream:
        list(provider.stream_single("Reply with exactly OK."))

    payload = mock_stream.call_args.kwargs["json"]
    assert "provider" not in payload


def test_openrouter_ask_with_tools_forwards_provider_preferences():
    """ask_with_tools threads the configured preferences into the shared loop."""
    prefs = {"order": ["Baidu", "Fireworks", "Alibaba"], "allow_fallbacks": True}
    config = ProviderConfig(
        api_key="test-key",
        model="qwen/qwen3.5-9b",
        provider_preferences=prefs,
    )
    provider = OpenRouterProvider(config)

    captured: dict = {}

    def fake_loop(**kwargs):
        captured.update(kwargs)
        return ("ok", [])

    with patch("cascade.providers.openrouter.openai_ask_with_tools", fake_loop):
        provider.ask_with_tools([{"role": "user", "content": "hi"}], tools={})

    assert captured["provider_preferences"] == prefs
