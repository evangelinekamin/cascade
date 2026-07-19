"""Tests for provider system."""

from cascade.providers.base import BaseProvider, ProviderConfig, Message


def test_provider_config_creation():
    """Test creating provider configuration."""
    config = ProviderConfig(
        api_key="test_key",
        model="test-model",
        temperature=0.5,
        max_tokens=100,
    )

    assert config.api_key == "test_key"
    assert config.model == "test-model"
    assert config.temperature == 0.5
    assert config.max_tokens == 100


def test_provider_config_defaults():
    """Test provider config with defaults."""
    config = ProviderConfig(
        api_key="key",
        model="model",
    )

    assert config.temperature == 0.7
    assert config.max_tokens is None
    assert config.base_url is None


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    def ask(self, messages, system=None):
        return f"Mock response to: {messages[-1]['content']}"

    def stream(self, messages, system=None):
        yield "Mock "
        yield "streaming "
        yield "response"

    def compare(self, prompt, system=None):
        return {
            "provider": self.name,
            "response": "Mock response",
            "length": 13,
        }


def test_mock_provider():
    """Test mock provider implementation."""
    config = ProviderConfig(api_key="key", model="mock")
    provider = MockProvider(config)

    assert provider.name == "MockProvider"
    assert provider.validate()

    # Test ask via convenience method
    response = provider.ask_single("test")
    assert "Mock response" in response

    # Test stream via convenience method
    streamed = "".join(provider.stream_single("test"))
    assert "Mock streaming response" in streamed

    # Test compare
    comparison = provider.compare("test")
    assert comparison["provider"] == "MockProvider"
    assert comparison["length"] == 13


def test_provider_receives_conversation_history():
    """Test that providers receive full conversation history."""
    config = ProviderConfig(api_key="key", model="mock")
    provider = MockProvider(config)

    messages: list[Message] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
    response = provider.ask(messages)
    assert "How are you?" in response


def test_provider_validation():
    """Test provider validation."""
    # Valid config
    valid_config = ProviderConfig(api_key="key", model="model")
    provider = MockProvider(valid_config)
    assert provider.validate()

    # Invalid config (missing api_key)
    invalid_config = ProviderConfig(api_key="", model="model")
    provider = MockProvider(invalid_config)
    assert not provider.validate()


def test_filter_activity():
    """Test _filter_activity strips activity prefixes and stores last."""
    config = ProviderConfig(api_key="key", model="mock")
    provider = MockProvider(config)

    raw_chunks = [
        "[[cascade_activity]] starting",
        "Hello",
        "[[cascade_activity]] model: test",
        " world",
    ]
    filtered = list(provider._filter_activity(iter(raw_chunks)))
    assert filtered == ["Hello", " world"]
    assert provider.last_activity == "model: test"


def test_filter_activity_dedupes_consecutive_normalized_updates():
    """Whitespace-only activity churn should not leak repeated updates."""
    config = ProviderConfig(api_key="key", model="mock")
    provider = MockProvider(config)

    raw_chunks = [
        "[[cascade_activity]]   thinking   hard   ",
        "[[cascade_activity]] thinking hard",
        "[[cascade_activity]] thinking    hard ",
        "Hello",
    ]

    filtered = list(provider._filter_activity(iter(raw_chunks)))

    assert filtered == ["Hello"]
    assert provider.last_activity == "thinking hard"


def test_base_provider_get_fallback_model_returns_none():
    """Test BaseProvider.get_fallback_model() returns None by default."""
    config = ProviderConfig(api_key="key", model="mock-model")
    provider = MockProvider(config)
    assert provider.get_fallback_model() is None


def test_gemini_get_fallback_model_pro_to_flash():
    """Test GeminiProvider.get_fallback_model() returns flash for pro models."""
    from cascade.providers.gemini import GeminiProvider

    config = ProviderConfig(api_key="fake-key", model="gemini-2.5-pro")
    provider = GeminiProvider(config)
    assert provider.get_fallback_model() == "gemini-2.5-flash"


def test_gemini_get_fallback_model_flash_returns_none():
    """Test GeminiProvider.get_fallback_model() returns None for flash models."""
    from cascade.providers.gemini import GeminiProvider

    config = ProviderConfig(api_key="fake-key", model="gemini-2.0-flash")
    provider = GeminiProvider(config)
    assert provider.get_fallback_model() is None


def test_claude_get_fallback_model_opus_to_sonnet():
    """Opus falls back to the current Sonnet; Sonnet itself has no fallback."""
    from cascade.providers.claude import ClaudeProvider

    config = ProviderConfig(api_key="fake-key", model="claude-sonnet-4-20250514")
    provider_sonnet = ClaudeProvider(config)
    assert provider_sonnet.get_fallback_model() is None

    # Any Opus falls back to the current Sonnet -- not a version-mangled string
    # like "claude-sonnet-4-8" (what naive opus->sonnet replacement would yield).
    for opus_model in ("claude-opus-4-20250514", "claude-opus-4-8"):
        provider_opus = ClaudeProvider(
            ProviderConfig(api_key="fake-key", model=opus_model)
        )
        assert provider_opus.get_fallback_model() == "claude-sonnet-5"


def test_openrouter_get_fallback_model_defaults_to_minimax():
    """OpenRouter should fall back to MiniMax when the primary model is overloaded."""
    from cascade.providers.openrouter import OpenRouterProvider

    config = ProviderConfig(api_key="fake-key", model="qwen/qwen3.5-9b")
    provider = OpenRouterProvider(config)
    assert provider.get_fallback_model() == "minimax/minimax-m2.5"


def test_condense_for_cli_single_message():
    """Test _condense_for_cli with one message returns just the content."""
    config = ProviderConfig(api_key="key", model="mock")
    provider = MockProvider(config)
    messages = [{"role": "user", "content": "Hello"}]
    assert provider._condense_for_cli(messages) == "Hello"


def test_condense_for_cli_with_history():
    """Test _condense_for_cli includes context from prior messages."""
    config = ProviderConfig(api_key="key", model="mock")
    provider = MockProvider(config)
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow up"},
    ]
    result = provider._condense_for_cli(messages)
    assert "Previous conversation context:" in result
    assert "User: First question" in result
    assert "Assistant: First answer" in result
    assert "Current request:\nFollow up" in result


def test_condense_for_cli_preserves_context_blocks_whole():
    """Episode/summary blocks are already compact -- never truncate them."""
    config = ProviderConfig(api_key="key", model="mock")
    provider = MockProvider(config)
    block = "[Prior session context]\n" + "episode line\n" * 300  # > per-msg cap
    messages = [
        {"role": "user", "content": block},
        {"role": "assistant", "content": "Understood, I have the episode context from prior interactions."},
        {"role": "user", "content": "Continue the work"},
    ]
    result = provider._condense_for_cli(messages)
    assert block in result
    assert "elided" not in result.split("Current request:")[0].split("episode line")[0]


def test_condense_for_cli_caps_messages_not_at_500():
    """Raw turns keep 2k chars each, not the old 500-char stubs."""
    config = ProviderConfig(api_key="key", model="mock")
    provider = MockProvider(config)
    long_answer = "a" * 1_500
    messages = [
        {"role": "assistant", "content": long_answer},
        {"role": "user", "content": "next"},
    ]
    result = provider._condense_for_cli(messages)
    assert long_answer in result  # under the 2k cap: fully preserved


def test_condense_for_cli_marks_elision_and_respects_budget():
    config = ProviderConfig(api_key="key", model="mock")
    provider = MockProvider(config)
    messages = [
        {"role": "assistant", "content": "b" * 10_000},
        {"role": "user", "content": "next"},
    ]
    result = provider._condense_for_cli(messages)
    assert "chars elided]" in result
    assert "b" * 2_001 not in result

    # 30 turns of 2k chars each exceeds the 24k budget: oldest are dropped
    many = [
        {"role": "user", "content": f"[turn {i}] " + "c" * 2_000}
        for i in range(30)
    ] + [{"role": "user", "content": "final request"}]
    result = provider._condense_for_cli(many)
    context = result.split("Current request:")[0]
    assert "[turn 29]" in context
    assert "[turn 0]" not in context
    assert result.endswith("final request")


def test_condense_for_cli_never_drops_context_blocks_under_budget_pressure():
    """Review-critical regression: the episode/summary block is the sole
    carrier of compacted history -- budget pressure must squeeze raw turns,
    never the block."""
    config = ProviderConfig(api_key="key", model="mock")
    provider = MockProvider(config)
    block = "[Prior session context]\n" + ("episode detail " * 2200)  # ~35k chars
    messages = [{"role": "user", "content": block}]
    messages += [
        {"role": "assistant", "content": f"[turn {i}] " + "x" * 1_800}
        for i in range(20)
    ]
    messages.append({"role": "user", "content": "final request"})

    result = provider._condense_for_cli(messages)
    context = result.split("Current request:")[0]
    assert "[Prior session context]" in context
    assert "episode detail" in context
    # Raw turns shrank to the floor but the newest still made it in
    assert "[turn 19]" in context
    assert result.endswith("final request")
