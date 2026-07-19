"""Tests for conversation history conversion and context window management."""

from unittest.mock import MagicMock

from cascade.conversation import (
    state_messages_to_provider,
    estimate_tokens,
    needs_compaction,
    compact_messages,
    compact_messages_with_episodes,
)
from cascade.state import CascadeState, ChatMessage


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


def test_live_same_provider_episodes_are_not_injected():
    """Live episodes mirror raw messages -- injecting them would double context."""
    from cascade.episodes import generate_episode

    messages = _msgs(
        ("you", "Fix the auth bug"),
        ("claude", "Fixed token validation."),
    )
    episodes = [generate_episode("Fix the auth bug", "Fixed token validation.", "claude")]
    result = state_messages_to_provider(
        messages, "claude", policy="summary", episodes=episodes,
    )
    contents = " ".join(m["content"] for m in result)
    assert "[Prior session context]" not in contents
    assert contents.count("Fixed token validation.") == 1


def test_live_other_provider_episodes_are_injected():
    """Cross-provider handoff: the raw window drops other providers' turns."""
    from cascade.episodes import generate_episode

    messages = _msgs(
        ("you", "Plan the refactor"),
        ("claude", "Plan: split the module."),
        ("you", "Implement it"),
    )
    episodes = [generate_episode("Plan the refactor", "Plan: split the module.", "claude")]
    result = state_messages_to_provider(
        messages, "openai", policy="summary", episodes=episodes,
    )
    contents = " ".join(m["content"] for m in result)
    assert "[Prior session context]" in contents
    assert "split the module" in contents
    # The raw claude message itself is not sent to openai under "summary"
    assert not any(m["content"] == "Plan: split the module." for m in result)


def test_compaction_episodes_always_injected():
    """Compaction episodes are the sole carrier of compacted-away turns."""
    from cascade.episodes import generate_episode

    messages = _msgs(("you", "Continue"))
    episodes = [
        generate_episode("Old task", "Old outcome.", "claude", source="compaction"),
    ]
    result = state_messages_to_provider(
        messages, "claude", policy="summary", episodes=episodes,
    )
    contents = " ".join(m["content"] for m in result)
    assert "[Prior session context]" in contents
    assert "Old task" in contents


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


def test_compacted_messages_are_excluded_from_provider_history():
    messages = [
        ChatMessage(role="you", content="old", metadata={"compacted": True}),
        ChatMessage(role="claude", content="old reply", metadata={"compacted": True}),
        ChatMessage(role="you", content="new"),
        ChatMessage(role="claude", content="new reply"),
    ]
    result = state_messages_to_provider(messages, "claude", policy="summary")
    assert result == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "new reply"},
    ]


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


def test_episode_compaction_marks_messages_so_it_does_not_repeat():
    state = CascadeState()
    state.messages = _msgs(
        ("you", "Task 1"),
        ("claude", "Done 1"),
        ("you", "Task 2"),
        ("claude", "Done 2"),
        ("you", "Task 3"),
        ("claude", "Done 3"),
        ("you", "Current"),
        ("claude", "Working"),
    )

    episodes, remaining = compact_messages_with_episodes(state.messages, keep_recent=2)
    active_count = len([m for m in state.messages if not m.metadata.get("compacted")])
    state.apply_episode_compaction(active_count - len(remaining), episodes)

    assert len(state.episodes) == 3

    later_episodes, later_remaining = compact_messages_with_episodes(state.messages, keep_recent=2)
    assert later_episodes == []
    assert [m.content for m in later_remaining] == ["Current", "Working"]


def test_full_policy_never_injects_live_episodes():
    """Under "full" every provider's raw turns are present -- live episodes
    would always duplicate."""
    from cascade.episodes import generate_episode

    messages = _msgs(
        ("you", "Plan it"),
        ("claude", "The plan."),
    )
    episodes = [generate_episode("Plan it", "The plan.", "claude")]
    result = state_messages_to_provider(
        messages, "openai", policy="full", episodes=episodes,
    )
    contents = " ".join(m["content"] for m in result)
    assert "[Prior session context]" not in contents
    assert contents.count("The plan.") == 1


def test_off_policy_restricts_episodes_to_own_provider():
    """"off" means no cross-model context, including via episodes."""
    from cascade.episodes import generate_episode

    messages = _msgs(("you", "Continue"))
    episodes = [
        generate_episode("Other work", "Done elsewhere.", "gemini", source="compaction"),
        generate_episode("My work", "Done here.", "claude", source="compaction"),
        generate_episode("[Compete] task", "Winner output.", "compete", source="orchestration"),
    ]
    result = state_messages_to_provider(
        messages, "claude", policy="off", episodes=episodes,
    )
    contents = " ".join(m["content"] for m in result)
    assert "Done here." in contents
    assert "Winner output." in contents
    assert "Done elsewhere." not in contents
