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
    """Cross-provider handoff: an older cross turn (beyond the sticky window)
    drops out of the raw window and is carried by its live episode instead.

    The most-recent cross turns ride verbatim (sticky), so only cross turns
    older than the sticky window rely on episode injection -- injecting a
    sticky turn's own episode would double it (see finding K)."""
    from cascade.episodes import generate_episode

    messages = _msgs(
        ("you", "Plan the refactor"),
        ("claude", "Plan: split the module."),   # older cross -> episode-only
        ("you", "review it"),
        ("gemini", "review pass one"),            # sticky cross (verbatim)
        ("openrouter", "review pass two"),        # sticky cross (verbatim)
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


class TestShouldCompact:
    """Pre-clip compaction trigger: token threshold OR raw-window overflow."""

    @staticmethod
    def _turns(n, content="hello world, a normal short message"):
        out = []
        for i in range(n):
            out.append(ChatMessage(role="you", content=f"[turn {i}] {content}"))
            out.append(ChatMessage(role="claude", content=f"[reply {i}] {content}"))
        return out

    def test_small_conversation_does_not_compact(self):
        from cascade.conversation import should_compact
        assert should_compact(self._turns(5), "claude") is False

    def test_more_than_raw_window_active_messages_compacts(self):
        """Beyond RAW_MESSAGE_WINDOW the clip would silently drop turns."""
        from cascade.conversation import should_compact
        assert should_compact(self._turns(21), "claude") is True  # 42 messages

    def test_compacted_messages_do_not_count_toward_the_window(self):
        from cascade.conversation import should_compact
        msgs = self._turns(21)
        for m in msgs[:10]:
            m.metadata["compacted"] = True
        assert should_compact(msgs, "claude") is False

    def test_anchor_occupancy_triggers_before_char_estimate_would(self):
        from cascade.conversation import should_compact
        from cascade.providers.usage import Usage

        msgs = self._turns(3)  # tiny tail
        anchor = Usage(input=100_000, output=2_000, cache_read=75_000)
        # claude window 200k -> threshold 171k; anchored total 177k exceeds it
        assert should_compact(msgs, "claude", anchor=anchor) is True
        assert should_compact(msgs, "claude") is False


def test_long_conversation_retains_old_context_via_compaction_episodes():
    """Regression: >40-message chats must not amnesia turns beyond the clip.

    Mirrors the worker sequence: should_compact -> episode compaction ->
    live-episode pruning -> payload build. The old topic must survive into
    the provider payload through the episode block.
    """
    from cascade.conversation import (
        compact_messages_with_episodes,
        should_compact,
        state_messages_to_provider,
    )
    from cascade.episodes import generate_episode, prune_live_episodes

    messages = []
    episodes = []
    for i in range(25):
        user = f"topic-{i}: please work on feature_{i}.py"
        reply = f"Done with feature_{i}.py implementation."
        messages.append(ChatMessage(role="you", content=user))
        messages.append(ChatMessage(role="claude", content=reply))
        episodes.append(generate_episode(user, reply, "claude"))

    assert should_compact(messages, "claude") is True

    new_episodes, remaining = compact_messages_with_episodes(messages, keep_recent=6)
    kept_exchanges = sum(1 for m in remaining if m.role == "you")
    episodes = prune_live_episodes(episodes, kept_exchanges) + new_episodes
    for m in messages[:-6]:
        m.metadata["compacted"] = True

    payload = state_messages_to_provider(
        messages, "claude", policy="summary", episodes=episodes,
    )
    contents = " ".join(m["content"] for m in payload)
    assert "[Prior session context]" in contents
    # Newest-first rendering under the char cap keeps recent compacted
    # topics; a mid-history topic well beyond the raw window must survive.
    assert "topic-20" in contents
    # And the kept raw tail is not duplicated by its own live episodes.
    assert contents.count("Done with feature_24.py implementation.") == 1


def test_prune_live_episodes_keeps_non_live_and_recent_live():
    from cascade.episodes import generate_episode, prune_live_episodes

    eps = [
        generate_episode("old live 1", "done", "claude"),
        generate_episode("compacted", "done", "claude", source="compaction"),
        generate_episode("old live 2", "done", "gemini"),
        generate_episode("recent live", "done", "claude"),
    ]
    pruned = prune_live_episodes(eps, keep_last=1)
    objectives = [ep.objective for ep in pruned]
    assert objectives == ["compacted", "recent live"]

    assert [ep.objective for ep in prune_live_episodes(eps, keep_last=0)] == ["compacted"]


class TestUnsentTail:
    def test_tail_counts_only_messages_after_last_provider_reply(self):
        from cascade.conversation import unsent_tail_chars

        msgs = _msgs(("you", "a" * 100), ("claude", "b" * 5000), ("you", "c" * 300))
        assert unsent_tail_chars(msgs) == 300

    def test_tail_zero_when_conversation_ends_with_provider(self):
        from cascade.conversation import unsent_tail_chars

        msgs = _msgs(("you", "a" * 100), ("claude", "b" * 5000))
        assert unsent_tail_chars(msgs) == 0

    def test_anchored_should_compact_does_not_double_count(self):
        """Review finding: anchor + whole-list estimate diverged ~2x from
        the displayed number and fired compaction early."""
        from cascade.conversation import should_compact
        from cascade.providers.usage import Usage

        # 160k tokens anchored on a 200k window (threshold 171k). The raw
        # transcript re-estimates to ~40k tokens; double-counting would put
        # occupancy at ~200k and fire. Correct accounting: 160k + tiny tail.
        msgs = _msgs(
            *[("you", "x" * 8000), ("claude", "y" * 8000)] * 10,
            ("you", "the new prompt"),
        )
        anchor = Usage(input=155_000, output=5_000)
        assert should_compact(msgs, "claude", anchor=anchor) is False
        near = Usage(input=168_000, output=4_000)
        assert should_compact(msgs, "claude", anchor=near) is True


def test_summary_keeps_recent_cross_provider_turns_verbatim():
    """A hand-off from another model (codex's found errors) must reach the model
    asked to act on it -- the summary policy used to drop it entirely."""
    messages = _msgs(
        ("you", "check the project"),
        ("openai", "FAIL: npm start hangs; FAIL: --include CACError; DepopHardRejectError"),
        ("you", "fix the errors codex found"),
    )
    result = state_messages_to_provider(messages, "openrouter", policy="summary")
    blob = "\n".join(m["content"] for m in result)
    # The openai (cross-provider) report is now inline + verbatim for openrouter.
    assert "DepopHardRejectError" in blob
    assert "[Response from openai]" in blob


def test_summary_only_keeps_the_last_two_cross_provider_turns():
    messages = _msgs(
        ("you", "q"),
        ("gemini", "OLD cross turn one"),
        ("claude", "OLD cross turn two"),
        ("openai", "RECENT cross turn A"),
        ("openai", "RECENT cross turn B"),
        ("you", "now you act, openrouter"),
    )
    result = state_messages_to_provider(messages, "openrouter", policy="summary")
    blob = "\n".join(m["content"] for m in result)
    assert "RECENT cross turn A" in blob and "RECENT cross turn B" in blob
    # The 3rd-most-recent cross turn falls back to episode-only (dropped here).
    assert "OLD cross turn one" not in blob


# --- System-role notices are visible to the model (not silently dropped) --------


def test_system_notice_is_emitted_under_summary_policy():
    messages = _msgs(
        ("you", "solve the auth bug"),
        ("system", "[Solve] fix the auth token validation"),
        ("you", "did that work?"),
    )
    result = state_messages_to_provider(messages, "openrouter", policy="summary")
    blob = "\n".join(m["content"] for m in result)
    assert "[System notice]" in blob
    assert "[Solve] fix the auth token validation" in blob


def test_system_notice_is_visible_even_under_off_policy():
    # "off" opts out of CROSS-MODEL chatter, but a session event is a fact about
    # what the user did, not another model's content, so it stays visible.
    messages = _msgs(
        ("you", "run it"),
        ("system", "[Fanout] build parser and printer"),
    )
    result = state_messages_to_provider(messages, "openrouter", policy="off")
    blob = "\n".join(m["content"] for m in result)
    assert "[Fanout] build parser and printer" in blob


def test_system_notice_pairs_with_an_assistant_ack():
    # Keeps user/assistant alternation intact for providers that require it.
    messages = _msgs(("system", "[Pipeline] migrate the schema"))
    result = state_messages_to_provider(messages, "openrouter", policy="summary")
    roles = [m["role"] for m in result]
    assert roles == ["user", "assistant"]
    assert result[1]["content"] == "Noted."


def test_non_orchestration_system_records_stay_ui_only():
    # A /tree dump, error, or separator recorded as a system message must NOT
    # leak into the model payload -- only [Solve]/[Pipeline]/[Fanout]/[Compete].
    messages = _msgs(
        ("you", "show me the tree"),
        ("system", "repo/\n  a.py\n  b.py\n  (full /tree dump)"),
        ("system", "Error: something failed in the UI"),
        ("you", "ok now build it"),
    )
    result = state_messages_to_provider(messages, "openrouter", policy="summary")
    blob = "\n".join(m["content"] for m in result)
    assert "full /tree dump" not in blob
    assert "something failed in the UI" not in blob
    assert "[System notice]" not in blob  # neither system row was surfaced
