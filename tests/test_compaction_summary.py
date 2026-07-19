"""Tier-2 compaction summary: prompt fidelity, validation, guards, injection.

The garbage-failure checklist as executable assertions: no pre-truncation,
originals never destroyed, invalid summaries never injected, summaries never
summarize themselves (previous summary merged as input), no-recap trailer.
"""

import pytest

from cascade.conversation import (
    SUMMARY_MIN_CHARS,
    build_compaction_summary_prompt,
    state_messages_to_provider,
    summarize_for_compaction,
    validate_compaction_summary,
)
from cascade.state import ChatMessage


def _msgs(n=10, size=1000):
    out = []
    for i in range(n):
        out.append(ChatMessage(role="you", content=f"[u{i}] " + "x" * size))
        out.append(ChatMessage(role="claude", content=f"[a{i}] " + "y" * size))
    return out


class TestPromptFidelity:
    def test_full_contents_no_per_message_truncation(self):
        """The old summarizer truncated messages to 1000 chars -- never again."""
        big = "z" * 5_000
        msgs = [
            ChatMessage(role="you", content="fix the bug in cascade/state.py"),
            ChatMessage(role="claude", content=big),
        ]
        prompt = build_compaction_summary_prompt(msgs)
        assert big in prompt  # verbatim, untruncated
        assert "cascade/state.py" in prompt

    def test_sections_and_verbatim_drift_guard(self):
        prompt = build_compaction_summary_prompt(_msgs(2))
        assert "Primary Request and Intent" in prompt
        assert "quote the most recent instructions verbatim" in prompt
        assert "All User Messages" in prompt

    def test_previous_summary_merged_not_resummarized(self):
        prompt = build_compaction_summary_prompt(
            _msgs(2), previous_summary="EARLIER: built the parser",
        )
        assert "merge" in prompt.lower()
        assert "EARLIER: built the parser" in prompt

    def test_oversized_transcript_drops_oldest_with_marker(self):
        msgs = _msgs(n=40, size=5_000)  # ~400k chars, way over the cap
        prompt = build_compaction_summary_prompt(msgs, max_input_chars=50_000)
        assert "[earlier turns truncated" in prompt
        # newest content survives, oldest dropped
        assert "[a39]" in prompt
        assert "[u0]" not in prompt


class TestValidation:
    def test_rejects_short_empty_and_error_shaped(self):
        assert validate_compaction_summary("") is False
        assert validate_compaction_summary("ok") is False
        assert validate_compaction_summary("Error: request failed" + "x" * 300) is False
        assert validate_compaction_summary("I cannot summarize this." + "x" * 300) is False

    def test_accepts_substantive_summary(self):
        good = "1. Primary Request and Intent\n" + "Real content. " * 30
        assert validate_compaction_summary(good) is True


class TestSummarizeGuards:
    def test_small_range_skipped_without_model_call(self):
        calls = []

        def ask(prompt, system):
            calls.append(prompt)
            return "x" * 500

        small = [ChatMessage(role="you", content="hi")]
        assert summarize_for_compaction(ask, small) is None
        assert calls == []

    def test_invalid_output_returns_none(self):
        def ask(prompt, system):
            return "Error: overloaded"

        assert summarize_for_compaction(ask, _msgs(5)) is None

    def test_overflow_failure_retries_with_reduced_range_and_visible_gap(self):
        calls = []

        def ask(prompt, system):
            calls.append(prompt)
            if len(calls) == 1:
                raise RuntimeError("request too large")
            return "1. Primary Request and Intent\n" + "Solid summary content. " * 20

        result = summarize_for_compaction(ask, _msgs(10))
        assert result is not None
        assert len(calls) == 2
        # Halved range, and the gap is surfaced to the summarizer explicitly
        assert "earlier turns were dropped" in calls[1]
        assert len(calls[1]) < len(calls[0])

    def test_transient_failure_retries_full_range(self):
        """A network blip must not amputate half the conversation."""
        calls = []

        def ask(prompt, system):
            calls.append(prompt)
            if len(calls) == 1:
                raise RuntimeError("connection reset by peer")
            return "1. Primary Request and Intent\n" + "Solid summary content. " * 20

        result = summarize_for_compaction(ask, _msgs(10))
        assert result is not None
        assert len(calls) == 2
        assert calls[1] == calls[0]
        assert "earlier turns were dropped" not in calls[1]

    def test_double_failure_returns_none(self):
        def ask(prompt, system):
            raise RuntimeError("down")

        assert summarize_for_compaction(ask, _msgs(10)) is None

    def test_originals_survive_regardless(self):
        """Summarization never mutates the messages it reads."""
        msgs = _msgs(5)
        before = [(m.role, m.content, dict(m.metadata)) for m in msgs]

        def ask(prompt, system):
            raise RuntimeError("boom")

        summarize_for_compaction(ask, msgs)
        after = [(m.role, m.content, dict(m.metadata)) for m in msgs]
        assert before == after


class TestInjection:
    def test_summary_injected_with_no_recap_trailer(self):
        msgs = [ChatMessage(role="you", content="continue")]
        result = state_messages_to_provider(
            msgs, "claude", policy="summary",
            compaction_summary="1. Primary Request: build the parser",
        )
        block = result[0]["content"]
        assert block.startswith("[Prior session context]")
        assert "build the parser" in block
        assert "Do not acknowledge, recap" in block
        assert result[1]["role"] == "assistant"

    def test_no_summary_no_block(self):
        msgs = [ChatMessage(role="you", content="continue")]
        result = state_messages_to_provider(msgs, "claude", policy="summary")
        assert len(result) == 1

    def test_summary_and_episodes_share_one_block(self):
        from cascade.episodes import generate_episode

        msgs = [ChatMessage(role="you", content="continue")]
        eps = [generate_episode("old task", "did it", "claude", source="compaction")]
        result = state_messages_to_provider(
            msgs, "claude", policy="summary",
            episodes=eps, compaction_summary="SUMMARY BODY " * 20,
        )
        blocks = [m for m in result if "[Prior session context]" in m.get("content", "")]
        assert len(blocks) == 1
        assert "SUMMARY BODY" in blocks[0]["content"]
        assert "old task" in blocks[0]["content"]


class TestPolicyGate:
    def test_summary_never_injected_under_policy_off(self):
        """policy 'off' opted out of cross-model context; the carried summary
        mixes providers by construction and must respect that."""
        msgs = [ChatMessage(role="you", content="continue")]
        result = state_messages_to_provider(
            msgs, "claude", policy="off",
            compaction_summary="gemini-derived content " * 20,
        )
        contents = " ".join(m["content"] for m in result)
        assert "gemini-derived" not in contents
        assert "[Prior session context]" not in contents
