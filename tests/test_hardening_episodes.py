"""Hardening tests for episode action provenance and Codex resume context.

Covers two reviewer findings:

(O) episodes._extract_actions must read the "tool" key that every real tool
    loop emits, not the never-populated "tool_name"/"name" keys.
(R) openai_provider synthetic-context detection must recognize a
    "[System notice]" user message so a resumed focused Codex session does
    not drop a surfaced orchestration notice -- including when the notice is
    preceded by a timeline marker like "[12:05]\n".
"""

from cascade.episodes import _extract_actions, generate_episode
from cascade.providers.openai_provider import OpenAIProvider


class TestEpisodeToolKey:
    """The real tool loops emit {"tool": name, ...}."""

    def test_reads_tool_key_first(self):
        tool_log = [
            {"tool": "read_file", "path": "a.py"},
            {"tool": "run_shell", "command": "pytest"},
        ]
        actions = _extract_actions("", tool_log)
        assert actions == ("tool:read_file", "tool:run_shell")

    def test_no_unknown_for_real_tool_records(self):
        tool_log = [{"tool": "write_file", "path": "x.py", "result": "ok"}]
        actions = _extract_actions("", tool_log)
        assert "tool:unknown" not in actions
        assert actions == ("tool:write_file",)

    def test_falls_back_to_legacy_keys(self):
        tool_log = [{"tool_name": "read_file"}, {"name": "bash"}]
        actions = _extract_actions("", tool_log)
        assert actions == ("tool:read_file", "tool:bash")

    def test_missing_all_keys_is_unknown(self):
        actions = _extract_actions("", [{"result": "ok"}])
        assert actions == ("tool:unknown",)

    def test_generate_episode_uses_tool_key(self):
        ep = generate_episode(
            user_content="Refactor the parser",
            assistant_content="Done.",
            provider="openai",
            tool_log=[{"tool": "edit_file", "path": "parser.py"}],
        )
        assert ep.actions == ("tool:edit_file",)


class TestSyntheticContextSystemNotice:
    """_has_synthetic_context must recognize the [System notice] tag."""

    def test_plain_system_notice_is_synthetic(self):
        messages = [
            {"role": "user", "content": "[System notice] escalation surfaced"},
            {"role": "user", "content": "Continue the audit."},
        ]
        assert OpenAIProvider._has_synthetic_context(messages) is True

    def test_system_notice_with_timeline_marker_is_synthetic(self):
        messages = [
            {"role": "user", "content": "[12:05]\n[System notice] escalation surfaced"},
            {"role": "user", "content": "Continue the audit."},
        ]
        assert OpenAIProvider._has_synthetic_context(messages) is True

    def test_ordinary_conversation_is_not_synthetic(self):
        messages = [
            {"role": "user", "content": "Audit frontend/src/lib/api.ts."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "Now rank the risks."},
        ]
        assert OpenAIProvider._has_synthetic_context(messages) is False

    def test_last_message_is_not_inspected(self):
        # Only prior messages count as synthetic context; the current request
        # is always the final message.
        messages = [
            {"role": "user", "content": "[System notice] surfaced"},
        ]
        assert OpenAIProvider._has_synthetic_context(messages) is False
