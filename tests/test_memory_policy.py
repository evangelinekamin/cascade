"""Tests for cross-model memory policy helpers."""

from cascade.cli import CascadeCore


def _make_app(policy: str = "summary") -> CascadeCore:
    app = CascadeCore.__new__(CascadeCore)
    app.memory_config = {
        "cross_model_memory": policy,
        "summary_turn_interval": 2,
        "summary_provider": "auto",
        "summary_max_chars": 1200,
    }
    app._conversation = []
    app._conversation_by_provider = {}
    app._cross_model_summary = ""
    app._last_provider_for_memory = None
    app._summary_turns_since_compact = 0
    return app


def test_summary_policy_compacts_on_provider_switch():
    app = _make_app("summary")
    app.record_turn("gemini", "Design a login flow", "We'll start with OAuth.")
    assert "Cross-model handoff summary" in app._cross_model_summary

    app.record_turn("claude", "Keep the same style", "We'll preserve prior constraints.")
    assert "Cross-model handoff summary" in app._cross_model_summary
    assert "Recent providers" in app._cross_model_summary


def test_summary_policy_builds_local_plus_summary_context():
    app = _make_app("summary")
    app.record_turn("gemini", "Use cyan accents", "Applied cyan + violet palette.")
    app.record_turn("claude", "Switch provider", "I'll continue from summary.")
    ctx = app._build_conversation_context("claude")
    assert "Cross-model handoff summary" in ctx
    assert "Current-provider recent turns" in ctx


def test_full_policy_uses_recent_transcript_without_summary():
    app = _make_app("full")
    app.record_turn("gemini", "A", "B")
    app.record_turn("claude", "C", "D")
    ctx = app._build_conversation_context("claude")
    assert "Conversation history (recent turns)" in ctx
    assert "User: A" in ctx
    assert "Assistant: B" in ctx
