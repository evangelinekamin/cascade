"""Tests for conversation history conversion and context window management."""

from unittest.mock import MagicMock

from cascade.conversation import (
    state_messages_to_provider,
    estimate_tokens,
    needs_compaction,
    compact_messages,
    CONTEXT_WINDOWS,
    COMPACTION_THRESHOLD,
)
from cascade.state import ChatMessage


def _msgs(*pairs):
    """Build ChatMessage list from (role, content) pairs."""
    return [ChatMessage(role=r, content=c) for r, c in pairs]


def test_off_policy_filters_to_target_provider():
    messages = _msgs(
        ("you", "Hello"),
        ("gemini", "Hi!"),
        ("you", "Switch"),
        ("claude", "Switched."),
        ("you", "Question"),
    )
    result = state_messages_to_provider(messages, "claude", policy="off")
    roles = [m["role"] for m in result]
    contents = [m["content"] for m in result]
    assert "Hi!" not in contents  # gemini response excluded
    assert "Switched." in contents
    assert roles.count("assistant") == 1


def test_summary_policy_injects_cross_model_summary():
    messages = _msgs(
        ("you", "Hello"),
        ("claude", "Hi!"),
    )
    result = state_messages_to_provider(
        messages, "claude", policy="summary",
        cross_model_summary="Previous context here.",
    )
    assert result[0]["content"].startswith("[Context from previous model interactions]")
    assert "Previous context here." in result[0]["content"]
    assert result[1]["role"] == "assistant"
    assert result[2]["role"] == "user"
    assert result[2]["content"] == "Hello"


def test_summary_policy_without_summary():
    messages = _msgs(
        ("you", "Hello"),
        ("claude", "Hi!"),
    )
    result = state_messages_to_provider(
        messages, "claude", policy="summary",
    )
    assert len(result) == 2
    assert result[0] == {"role": "user", "content": "Hello"}
    assert result[1] == {"role": "assistant", "content": "Hi!"}


def test_full_policy_includes_cross_provider_messages():
    messages = _msgs(
        ("you", "Hello"),
        ("gemini", "Gemini says hi."),
        ("you", "Now ask claude"),
        ("claude", "Claude here."),
    )
    result = state_messages_to_provider(messages, "claude", policy="full")
    # Gemini response should appear as context
    contents = " ".join(m["content"] for m in result)
    assert "[Response from gemini]" in contents
    assert "Gemini says hi." in contents
    assert "Claude here." in contents


def test_char_budget_trims_oldest():
    messages = _msgs(
        ("you", "A" * 1000),
        ("claude", "B" * 1000),
        ("you", "C" * 1000),
        ("claude", "D" * 1000),
    )
    result = state_messages_to_provider(
        messages, "claude", policy="summary", max_chars=2500,
    )
    total = sum(len(m["content"]) for m in result)
    assert total <= 2500
    # Most recent messages should survive
    assert result[-1]["content"] == "D" * 1000


def test_empty_messages():
    result = state_messages_to_provider([], "claude", policy="summary")
    assert result == []


def test_message_format_user_role():
    messages = _msgs(("you", "Hello"))
    result = state_messages_to_provider(messages, "claude", policy="full")
    assert result == [{"role": "user", "content": "Hello"}]


# ------------------------------------------------------------------
# Phase 5: Context window management
# ------------------------------------------------------------------


def test_estimate_tokens():
    messages = [
        {"role": "user", "content": "A" * 400},
        {"role": "assistant", "content": "B" * 800},
    ]
    # 1200 chars / 4 = 300 tokens
    assert estimate_tokens(messages) == 300


def test_estimate_tokens_empty():
    assert estimate_tokens([]) == 0


def test_needs_compaction_under_threshold():
    # 100 chars / 4 = 25 tokens, well under any window
    messages = [{"role": "user", "content": "x" * 100}]
    assert needs_compaction(messages, "claude") is False


def test_needs_compaction_over_threshold():
    # claude window = 200_000, threshold = 0.75 -> 150_000 tokens -> 600_000 chars
    big_content = "x" * 700_000
    messages = [{"role": "user", "content": big_content}]
    assert needs_compaction(messages, "claude") is True


def test_needs_compaction_unknown_provider_uses_default():
    # unknown provider defaults to 128_000 tokens -> threshold 96_000 -> 384_000 chars
    messages = [{"role": "user", "content": "x" * 400_000}]
    assert needs_compaction(messages, "unknown_provider") is True


def test_compact_messages_short_history_unchanged():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    mock_provider = MagicMock()
    result = compact_messages(messages, mock_provider, keep_recent=6)
    assert result == messages
    mock_provider.ask_single.assert_not_called()


def test_compact_messages_summarizes_old_keeps_recent():
    messages = [
        {"role": "user", "content": f"msg_{i}"}
        for i in range(10)
    ]
    mock_provider = MagicMock()
    mock_provider.ask_single.return_value = "Summary of earlier conversation."

    result = compact_messages(messages, mock_provider, keep_recent=4)

    # Should have: summary user msg + ack + 4 recent = 6 messages
    assert len(result) == 6
    assert "[Conversation summary]" in result[0]["content"]
    assert "Summary of earlier conversation." in result[0]["content"]
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "Understood, I have the context. Continuing."
    # Last 4 messages preserved
    for i, msg in enumerate(result[2:]):
        assert msg["content"] == f"msg_{i + 6}"

    mock_provider.ask_single.assert_called_once()


def test_compact_messages_truncates_old_content_in_transcript():
    long_content = "x" * 2000
    messages = [
        {"role": "user", "content": long_content},
        {"role": "assistant", "content": long_content},
        {"role": "user", "content": "recent1"},
        {"role": "assistant", "content": "recent2"},
    ]
    mock_provider = MagicMock()
    mock_provider.ask_single.return_value = "Summary."

    compact_messages(messages, mock_provider, keep_recent=2)

    # The transcript passed to ask_single should have truncated content
    call_args = mock_provider.ask_single.call_args
    prompt = call_args[0][0]
    # Each old message content is truncated to 1000 chars
    assert "x" * 1001 not in prompt
