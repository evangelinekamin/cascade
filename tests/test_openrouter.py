"""Tests for the OpenRouter provider."""

from unittest.mock import patch

import httpx
import pytest

from cascade.providers.base import ProviderConfig
from cascade.providers.openrouter import OpenRouterProvider
from cascade.providers.usage import Usage


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
    assert provider.last_usage == Usage(input=7, output=2)
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


def test_openrouter_stream_sends_session_id_for_sticky_cache():
    """The streaming payload carries a session_id so OpenRouter pins the
    conversation to one host and keeps its prompt cache warm."""
    provider = OpenRouterProvider(ProviderConfig(api_key="test-key", model="m"))

    with patch.object(provider.client, "stream", return_value=_ok_stream_context()) as mock_stream:
        list(provider.stream_single("Reply with exactly OK."))

    payload = mock_stream.call_args.kwargs["json"]
    assert payload["session_id"].startswith("cascade-")
    assert len(payload["session_id"]) <= 256


def test_stream_omits_temperature_for_a_reasoning_model():
    """When the model's endpoints reject temperature, it must not be sent --
    otherwise require_parameters filters out every endpoint (a 404)."""
    provider = OpenRouterProvider(ProviderConfig(api_key="k", model="openai/gpt-5.6-luna"))
    provider._send_temperature = False  # what _apply_meta sets for a reasoning model

    with patch.object(provider.client, "stream", return_value=_ok_stream_context()) as mock_stream:
        list(provider.stream_single("hi"))

    assert "temperature" not in mock_stream.call_args.kwargs["json"]


def test_http_error_text_surfaces_openrouter_body_reason():
    """A 404's real reason lives in the JSON body, not httpx's status line."""
    from cascade.providers._openai_tools import _http_error_text

    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = httpx.Response(
        404,
        json={"error": {
            "message": "No endpoints found matching your data policy",
            "metadata": {"error_type": "no_endpoints"},
        }},
        request=req,
    )
    msg = _http_error_text(httpx.HTTPStatusError("404", request=req, response=resp))
    assert "404" in msg
    assert "No endpoints found matching your data policy" in msg
    assert "no_endpoints" in msg


def test_http_error_text_falls_back_to_status_line_without_body():
    from cascade.providers._openai_tools import _http_error_text

    req = httpx.Request("POST", "https://example.test/x")
    resp = httpx.Response(500, text="", request=req)
    msg = _http_error_text(httpx.HTTPStatusError("500 Server Error", request=req, response=resp))
    assert "500" in msg


def test_session_key_is_stable_across_turns_and_unique_per_conversation():
    key = OpenRouterProvider._session_key
    system = "You are Cascade. cwd=/x mode=plan"
    turn1 = [{"role": "user", "content": "plan it", "provider": ""}]
    # A later turn appends assistant + user messages; the stable prefix is
    # unchanged, so the key must not move (same warm cache).
    turn2 = turn1 + [
        {"role": "assistant", "content": "a plan", "provider": "openrouter"},
        {"role": "user", "content": "now add auth", "provider": ""},
    ]
    assert key(system, turn1) == key(system, turn2)
    # A different opening message is a different conversation.
    assert key(system, [{"role": "user", "content": "other", "provider": ""}]) != key(system, turn1)
    # Nothing to key on -> no id (so the field is simply omitted).
    assert key(None, []) is None


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


def test_openrouter_midstream_error_is_not_treated_as_success():
    """HTTP 200 can still contain an in-band provider failure after partial text."""
    config = ProviderConfig(api_key="test-key", model="test-model")
    provider = OpenRouterProvider(config)

    class _Response:
        headers = {"X-Generation-Id": "gen-partial"}

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}'
            yield (
                'data: {"error":{"code":429,"message":"overloaded",'
                '"metadata":{"error_type":"provider_overloaded"}},'
                '"choices":[{"delta":{"content":""},"finish_reason":"error"}]}'
            )

    class _Context:
        def __enter__(self):
            return _Response()

        def __exit__(self, exc_type, exc, tb):
            return False

    with patch.object(provider.client, "stream", return_value=_Context()):
        stream = provider.stream_single("hello")
        assert next(stream) == "partial"
        with pytest.raises(RuntimeError, match="provider_overloaded"):
            next(stream)

    assert provider.last_generation_id == "gen-partial"


def test_openrouter_structured_response_uses_schema_and_required_provider_params():
    prefs = {
        "order": ["cerebras"],
        "allow_fallbacks": True,
        "require_parameters": True,
    }
    provider = OpenRouterProvider(
        ProviderConfig(
            api_key="test-key",
            model="openai/gpt-oss-120b",
            provider_preferences=prefs,
        )
    )
    response = httpx.Response(
        200,
        headers={"X-Generation-Id": "gen-route"},
        json={
            "choices": [{
                "message": {
                    "content": '{"workflow":"recon","reason":"read only","confidence":0.9}'
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "cost": 0.001},
        },
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )
    schema = {
        "type": "object",
        "properties": {"workflow": {"type": "string"}},
        "required": ["workflow"],
    }

    with patch.object(provider.client, "post", return_value=response) as post:
        result = provider.ask_structured("route this", schema, schema_name="route")

    payload = post.call_args.kwargs["json"]
    assert result["workflow"] == "recon"
    assert payload["model"] == "openai/gpt-oss-120b"
    assert payload["provider"] == prefs
    assert payload["response_format"]["json_schema"] == {
        "name": "route",
        "strict": True,
        "schema": schema,
    }
    assert provider.last_usage == Usage(input=12, output=5, cost=0.001)
    assert provider.last_cost == 0.001
    assert provider.last_generation_id == "gen-route"


def test_openrouter_tool_loop_captures_usage_cost():
    provider = OpenRouterProvider(
        ProviderConfig(api_key="test-key", model="openai/gpt-oss-120b")
    )
    response = httpx.Response(
        200,
        json={
            "choices": [{
                "message": {"content": "done"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 4, "cost": 0.0025},
        },
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )

    with patch.object(provider.client, "post", return_value=response):
        text, log = provider.ask_with_tools(
            [{"role": "user", "content": "inspect"}],
            {},
        )

    assert text == "done"
    assert log == []
    assert provider.last_usage == Usage(input=20, output=4, cost=0.0025)
    assert provider.last_cost == pytest.approx(0.0025)
