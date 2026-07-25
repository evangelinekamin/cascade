"""Tests for the episode-based compaction system."""

from cascade.episodes import (
    Episode,
    generate_episode,
    episodes_to_context,
    compact_to_episodes,
    _extract_objective,
    _extract_artifacts,
    _extract_actions,
    _extract_outcome,
)
from cascade.state import ChatMessage


class TestEpisodeGeneration:
    """Tests for generating episodes from interactions."""

    def test_basic_episode(self):
        ep = generate_episode(
            user_content="Fix the bug in auth.py",
            assistant_content="I fixed the bug by updating the token validation.",
            provider="claude",
        )
        assert isinstance(ep, Episode)
        assert ep.provider == "claude"
        assert "auth.py" in ep.objective
        assert ep.raw_turn_count == 1

    def test_episode_with_tools(self):
        tool_log = [
            {"tool_name": "read_file", "result": "ok"},
            {"tool_name": "write_file", "result": "ok"},
        ]
        ep = generate_episode(
            user_content="Update the config",
            assistant_content="Done.",
            provider="openai",
            tokens=1500,
            tool_log=tool_log,
        )
        assert "tool:read_file" in ep.actions
        assert "tool:write_file" in ep.actions
        assert ep.tokens_consumed == 1500

    def test_episode_with_turn_count(self):
        ep = generate_episode(
            user_content="Multi-turn task",
            assistant_content="Completed in 3 rounds.",
            provider="gemini",
            turn_count=3,
        )
        assert ep.raw_turn_count == 3

    def test_episode_id_is_deterministic_per_call(self):
        ep1 = generate_episode("a", "b", "claude")
        ep2 = generate_episode("a", "b", "claude")
        # Different timestamps -> different ids
        assert ep1.id != ep2.id

    def test_episode_is_frozen(self):
        ep = generate_episode("a", "b", "claude")
        try:
            ep.provider = "gemini"
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestObjectiveExtraction:
    """Tests for extracting objectives from user prompts."""

    def test_single_line(self):
        assert _extract_objective("Fix the login bug") == "Fix the login bug"

    def test_multiline_takes_first(self):
        text = "Fix the login bug\nAlso update the tests\nAnd the docs"
        assert _extract_objective(text) == "Fix the login bug"

    def test_skips_code_fences(self):
        text = "```python\ncode\n```\nActual objective"
        assert _extract_objective(text) == "Actual objective"

    def test_truncation(self):
        long_text = "x" * 300
        result = _extract_objective(long_text, max_len=100)
        assert len(result) <= 104  # 100 + "..."


class TestArtifactExtraction:
    """Tests for extracting file paths from text."""

    def test_python_files(self):
        text = "I modified cascade/cli.py and cascade/state.py"
        artifacts = _extract_artifacts(text)
        assert "cascade/cli.py" in artifacts
        assert "cascade/state.py" in artifacts

    def test_path_like_tokens(self):
        text = "Check src/components/Button.tsx for the issue"
        artifacts = _extract_artifacts(text)
        assert "src/components/Button.tsx" in artifacts

    def test_strips_markdown(self):
        text = "Modified `cascade/hooks/runner.py` file"
        artifacts = _extract_artifacts(text)
        assert "cascade/hooks/runner.py" in artifacts

    def test_max_artifacts(self):
        text = " ".join(f"file{i}.py" for i in range(20))
        artifacts = _extract_artifacts(text, max_artifacts=5)
        assert len(artifacts) <= 5

    def test_ignores_urls(self):
        text = "See https://example.com/path/to/file.py for details"
        artifacts = _extract_artifacts(text)
        assert not any("https" in a for a in artifacts)


class TestActionExtraction:
    """Tests for extracting actions from responses."""

    def test_tool_actions(self):
        tool_log = [{"tool_name": "read_file"}, {"name": "bash"}]
        actions = _extract_actions("", tool_log)
        assert "tool:read_file" in actions
        assert "tool:bash" in actions

    def test_heuristic_created(self):
        text = "I created utils/helper.py with the new function"
        actions = _extract_actions(text)
        assert any("created:" in a for a in actions)

    def test_heuristic_modified(self):
        text = "Modified the configuration in config.yaml"
        actions = _extract_actions(text)
        assert any("modified:" in a for a in actions)

    def test_empty_response(self):
        actions = _extract_actions("")
        assert actions == ()


class TestOutcomeExtraction:
    """Tests for extracting outcomes from responses."""

    def test_single_paragraph(self):
        text = "The bug was caused by a missing null check."
        assert _extract_outcome(text) == text

    def test_prefers_last_non_code_paragraph(self):
        text = "First paragraph.\n\n```python\ncode\n```\n\nThe fix is deployed."
        outcome = _extract_outcome(text)
        assert "fix is deployed" in outcome

    def test_truncation(self):
        long_text = "x" * 500
        result = _extract_outcome(long_text, max_len=100)
        assert len(result) <= 104

    def test_prefers_failure_lines_over_a_generic_closing_paragraph(self):
        text = (
            "I ran the suite.\n\n"
            "FAILED tests/test_stdio.py::test_handshake - Connection closed\n"
            "3 failed, 340 passed\n\n"
            "Let me know if you'd like me to keep digging."
        )
        outcome = _extract_outcome(text)
        assert "FAILED tests/test_stdio.py::test_handshake" in outcome
        assert "3 failed" in outcome
        assert "keep digging" not in outcome  # the generic sign-off is not the outcome

    def test_code_fence_is_not_scanned_for_failures(self):
        # An example with `except Exception:` inside a fence is not a failure.
        text = (
            "Here's the pattern:\n\n"
            "```python\ntry:\n    run()\nexcept Exception:\n    log()\n```\n\n"
            "All 340 tests pass now."
        )
        assert _extract_outcome(text) == "All 340 tests pass now."

    def test_zero_failures_and_fixed_errors_are_not_flagged(self):
        # Success summaries that mention counts/exceptions must NOT read as failure.
        assert _extract_outcome("Ran it.\n\n340 passed, 0 failed in 12s") == "340 passed, 0 failed in 12s"
        assert _extract_outcome("Done.\n\nAdded Exception handling; fixed 3 errors.") == "Added Exception handling; fixed 3 errors."

    def test_success_prose_is_untouched_by_the_failure_scan(self):
        text = "I added error handling and the tests pass with no errors.\n\nAll green."
        assert _extract_outcome(text) == "All green."

    def test_failure_outcome_is_bounded(self):
        text = "\n".join(f"FAILED test_{i} - boom because reasons" for i in range(50))
        assert len(_extract_outcome(text, max_len=200)) <= 204


class TestEpisodesToContext:
    """Tests for rendering episodes as model context."""

    def test_empty_episodes(self):
        assert episodes_to_context([]) == ""

    def test_single_episode(self):
        ep = generate_episode("Fix bug", "Fixed it.", "claude", tokens=100)
        context = episodes_to_context([ep])
        assert "Episode history" in context
        assert "Fix bug" in context
        assert "Fixed it." in context

    def test_max_chars_limit(self):
        episodes = [
            generate_episode(f"Task {i}", f"Result {i}" * 100, "claude")
            for i in range(20)
        ]
        context = episodes_to_context(episodes, max_chars=500)
        assert len(context) < 800  # some overhead for headers

    def test_chronological_order(self):
        ep1 = Episode(
            id="aaa", timestamp=1.0, provider="claude",
            objective="First", actions=(), outcome="Done",
            artifacts=(), tokens_consumed=0, raw_turn_count=1,
        )
        ep2 = Episode(
            id="bbb", timestamp=2.0, provider="gemini",
            objective="Second", actions=(), outcome="Done",
            artifacts=(), tokens_consumed=0, raw_turn_count=1,
        )
        context = episodes_to_context([ep1, ep2])
        assert context.index("First") < context.index("Second")


class TestCompactToEpisodes:
    """Tests for converting messages to episodes."""

    def _msgs(self, *pairs):
        return [ChatMessage(role=r, content=c) for r, c in pairs]

    def test_short_history_no_episodes(self):
        messages = self._msgs(
            ("you", "Hello"),
            ("claude", "Hi!"),
        )
        episodes, recent = compact_to_episodes(messages, keep_recent=6)
        assert episodes == []
        assert len(recent) == 2

    def test_compacts_old_messages(self):
        messages = self._msgs(
            ("you", "Task 1"),
            ("claude", "Done 1"),
            ("you", "Task 2"),
            ("gemini", "Done 2"),
            ("you", "Task 3"),
            ("claude", "Done 3"),
            ("you", "Current task"),
            ("claude", "Working on it"),
        )
        episodes, recent = compact_to_episodes(messages, keep_recent=4)
        assert len(episodes) == 2  # Two old user-assistant pairs
        assert len(recent) == 4
        assert recent[0].content == "Task 3"

    def test_episode_preserves_provider(self):
        messages = self._msgs(
            ("you", "Question"),
            ("gemini", "Answer from Gemini"),
            ("you", "Recent"),
            ("claude", "Recent answer"),
        )
        episodes, recent = compact_to_episodes(messages, keep_recent=2)
        assert len(episodes) == 1
        assert episodes[0].provider == "gemini"

    def test_orphan_assistant_message(self):
        messages = self._msgs(
            ("claude", "Orphan response"),
            ("you", "Recent"),
            ("claude", "Recent answer"),
        )
        episodes, recent = compact_to_episodes(messages, keep_recent=2)
        assert len(episodes) == 1
        assert episodes[0].provider == "claude"

    def test_all_recent_no_episodes(self):
        messages = self._msgs(
            ("you", "A"),
            ("claude", "B"),
        )
        episodes, recent = compact_to_episodes(messages, keep_recent=10)
        assert episodes == []
        assert len(recent) == 2


class TestConversationWithEpisodes:
    """Tests for episode integration in conversation.py."""

    def test_state_messages_with_compaction_episodes(self):
        from cascade.conversation import state_messages_to_provider

        episodes = [
            generate_episode(
                "Fix auth", "Fixed token validation.", "claude",
                source="compaction",
            ),
        ]
        messages = [
            ChatMessage(role="you", content="Now update tests"),
            ChatMessage(role="claude", content="Tests updated."),
        ]
        result = state_messages_to_provider(
            messages, "claude", policy="summary",
            episodes=episodes,
        )
        # Should have episode context + regular messages
        assert any("[Prior session context]" in m["content"] for m in result)
        assert any("Fix auth" in m["content"] for m in result)
        assert any("Now update tests" in m["content"] for m in result)

    def test_live_episode_for_same_provider_is_filtered(self):
        from cascade.conversation import state_messages_to_provider

        episodes = [
            generate_episode("Task", "Done.", "claude"),
        ]
        messages = [
            ChatMessage(role="you", content="Hello"),
        ]
        result = state_messages_to_provider(
            messages, "claude", policy="summary",
            episodes=episodes,
        )
        contents = " ".join(m["content"] for m in result)
        # A live episode mirrors raw turns of the same provider: not injected
        assert "Episode" not in contents

    def test_no_episodes_sends_raw_messages_only(self):
        from cascade.conversation import state_messages_to_provider

        messages = [
            ChatMessage(role="you", content="Hello"),
        ]
        result = state_messages_to_provider(
            messages, "claude", policy="summary",
            episodes=None,
        )
        assert result == [{"role": "user", "content": "Hello"}]
