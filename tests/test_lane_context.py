"""build_lane_context: the bounded conversation digest fed into focused lanes.

A recon/solve/pipeline/fanout lane runs one task prompt in a fresh worktree with
no chat history, so a referential request ("fix the errors codex found") reaches
it with no trace of its referent. build_lane_context carries the minimum prior
context that makes such a request resolvable, without becoming a full transcript.
"""

from cascade.conversation import build_lane_context, _LANE_CONTEXT_MAX_CHARS
from cascade.state import ChatMessage


def _m(role, content, metadata=None):
    return ChatMessage(role=role, content=content, metadata=metadata or {})


def test_empty_history_yields_no_context():
    assert build_lane_context([]) == ""


def test_only_user_turns_with_no_report_still_frames_the_task():
    hist = [_m("you", "start something"), _m("you", "keep going")]
    ctx = build_lane_context(hist, "openrouter")
    assert "keep going" in ctx
    assert "for reference only" in ctx  # wrapped as reference, not instruction


def test_carries_the_last_cross_provider_report_verbatim():
    hist = [
        _m("you", "add a feature"),
        _m("openrouter", "feature added"),
        _m("you", "have codex review it"),
        _m("codex", "I found 3 errors: A, B, C in server.ts"),
    ]
    ctx = build_lane_context(hist, "openrouter")
    assert "codex: I found 3 errors: A, B, C in server.ts" in ctx


def test_carries_both_cross_and_same_provider_reports():
    # "fix what codex found" (cross) AND "apply what you suggested" (same) both
    # have a live referent, so the digest keeps the most recent of each.
    hist = [
        _m("openrouter", "SAME_PROVIDER_PLAN: refactor the parser"),
        _m("you", "first, ask codex"),
        _m("codex", "CROSS_PROVIDER_REPORT: three bugs"),
        _m("you", "ok"),
    ]
    ctx = build_lane_context(hist, "openrouter")
    assert "CROSS_PROVIDER_REPORT" in ctx
    assert "SAME_PROVIDER_PLAN" in ctx


def test_compacted_messages_are_excluded():
    hist = [
        _m("codex", "OLD_COMPACTED_REPORT", metadata={"compacted": True}),
        _m("you", "the recent ask"),
        _m("gemini", "RECENT_REPORT visible"),
    ]
    ctx = build_lane_context(hist, "openrouter")
    assert "OLD_COMPACTED_REPORT" not in ctx
    assert "RECENT_REPORT" in ctx


def test_recent_user_turns_are_bounded():
    # Only the last few user turns are framed; older ones drop out.
    hist = [_m("you", f"turn number {i}") for i in range(10)]
    ctx = build_lane_context(hist, "openrouter", max_user_turns=3)
    assert "turn number 9" in ctx
    assert "turn number 8" in ctx
    assert "turn number 0" not in ctx


def test_long_report_is_clipped_and_output_stays_within_budget():
    huge = "HEAD_MARKER " + ("x" * 20000) + " TAIL_MARKER"
    hist = [_m("you", "look at this"), _m("codex", huge)]
    ctx = build_lane_context(hist, "openrouter")
    # Bounded overall...
    assert len(ctx) <= _LANE_CONTEXT_MAX_CHARS + 400  # +wrapper header
    # ...but keeps both ends of the report (middle elided).
    assert "HEAD_MARKER" in ctx
    assert "TAIL_MARKER" in ctx
    assert "[...]" in ctx


def test_current_turn_is_the_callers_responsibility_to_slice():
    # The helper digests exactly what it is given; the live turn is excluded by
    # the caller slicing history[:-1], which this test mirrors.
    full = [
        _m("codex", "the referent report"),
        _m("you", "fix the errors codex found"),  # the live turn
    ]
    ctx = build_lane_context(full[:-1], "openrouter")
    assert "the referent report" in ctx
    assert "fix the errors codex found" not in ctx


def test_chronological_order_is_preserved():
    hist = [
        _m("you", "AAA first"),
        _m("codex", "BBB middle report"),
        _m("you", "CCC last"),
    ]
    ctx = build_lane_context(hist, "openrouter")
    assert ctx.index("AAA") < ctx.index("BBB") < ctx.index("CCC")
