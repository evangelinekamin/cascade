"""Hardening tests for cascade.conversation (4-reviewer findings I, J, K, N).

I  build_lane_context honors cross_model_memory "off" (no other-model reports).
J  build_lane_context recovers a referent lost to compaction from episodes/summary.
K  sticky cross turns are not double-emitted (verbatim + episode) and are clipped.
N  the char-budget trim removes synthetic (user, "Noted.") pairs atomically.
"""

from cascade.conversation import (
    _STICKY_CROSS_MAX_CHARS,
    build_lane_context,
    state_messages_to_provider,
)
from cascade.episodes import Episode, generate_episode
from cascade.state import ChatMessage


def _m(role, content, metadata=None):
    return ChatMessage(role=role, content=content, metadata=metadata or {})


def _msgs(*pairs):
    return [ChatMessage(role=r, content=c) for r, c in pairs]


# --------------------------------------------------------------------------- #
# (I) build_lane_context honors policy="off"
# --------------------------------------------------------------------------- #

def test_off_policy_excludes_cross_provider_reports():
    hist = [
        _m("you", "add a feature"),
        _m("openrouter", "SAME_PROVIDER_PLAN done"),
        _m("you", "have codex review it"),
        _m("codex", "CROSS_PROVIDER_REPORT: three bugs"),
    ]
    ctx = build_lane_context(hist, "openrouter", policy="off")
    # Same-provider report and user turns survive; the other model's does not.
    assert "SAME_PROVIDER_PLAN" in ctx
    assert "have codex review it" in ctx
    assert "CROSS_PROVIDER_REPORT" not in ctx


def test_off_policy_never_falls_back_to_a_cross_report():
    # No same-provider turn exists; "off" must NOT substitute the cross report
    # via the any-report fallback -- only user turns remain.
    hist = [
        _m("you", "kick things off"),
        _m("codex", "CROSS_ONLY_REPORT: found an issue"),
        _m("you", "now act on it"),
    ]
    ctx = build_lane_context(hist, "openrouter", policy="off")
    assert "CROSS_ONLY_REPORT" not in ctx
    assert "now act on it" in ctx


def test_summary_policy_default_still_carries_cross_reports():
    # Default preserves today's behavior: the cross report is the referent.
    hist = [
        _m("you", "have codex review it"),
        _m("codex", "CROSS_PROVIDER_REPORT: three bugs"),
    ]
    ctx = build_lane_context(hist, "openrouter")
    assert "CROSS_PROVIDER_REPORT" in ctx


# --------------------------------------------------------------------------- #
# (J) build_lane_context recovers a compaction-lost referent
# --------------------------------------------------------------------------- #

def test_recovers_referent_from_episode_when_report_compacted():
    # The cross report was compacted out; only user turns remain visible. The
    # episode outcome carrying it must be surfaced so "fix it" still resolves.
    hist = [
        _m("codex", "COMPACTED_REPORT: bug in parser", metadata={"compacted": True}),
        _m("you", "fix what codex found"),
    ]
    ep = Episode(
        id="e1", timestamp=1.0, provider="codex", objective="review",
        actions=(), outcome="RECOVERED_OUTCOME: bug in parser at line 42",
        artifacts=(), tokens_consumed=0, raw_turn_count=2, source="live",
    )
    ctx = build_lane_context(hist, "openrouter", episodes=[ep])
    assert "RECOVERED_OUTCOME" in ctx
    assert "Compacted earlier context" in ctx
    assert "fix what codex found" in ctx


def test_recovers_referent_from_compaction_summary_when_no_episode():
    hist = [
        _m("codex", "COMPACTED_REPORT", metadata={"compacted": True}),
        _m("you", "act on it"),
    ]
    summary = "SUMMARY_REFERENT: codex reported a null-deref in server.ts"
    ctx = build_lane_context(hist, "openrouter", compaction_summary=summary)
    assert "SUMMARY_REFERENT" in ctx
    assert "Compacted earlier context" in ctx


def test_recovery_prefers_episode_over_summary():
    hist = [_m("you", "act on it")]
    ep = Episode(
        id="e1", timestamp=1.0, provider="codex", objective="o",
        actions=(), outcome="EPISODE_WINS", artifacts=(),
        tokens_consumed=0, raw_turn_count=1, source="live",
    )
    ctx = build_lane_context(
        hist, "openrouter", episodes=[ep], compaction_summary="SUMMARY_LOSES",
    )
    assert "EPISODE_WINS" in ctx
    assert "SUMMARY_LOSES" not in ctx


def test_recovery_is_skipped_when_a_visible_report_survives():
    # A live report still present -- no recovery excerpt is appended.
    hist = [
        _m("you", "review it"),
        _m("codex", "VISIBLE_REPORT here"),
    ]
    ep = Episode(
        id="e1", timestamp=1.0, provider="codex", objective="o",
        actions=(), outcome="STALE_EPISODE", artifacts=(),
        tokens_consumed=0, raw_turn_count=1, source="live",
    )
    ctx = build_lane_context(hist, "openrouter", episodes=[ep])
    assert "VISIBLE_REPORT" in ctx
    assert "STALE_EPISODE" not in ctx


def test_off_policy_recovery_ignores_cross_provider_episode_and_summary():
    # Under "off", recovery must not surface another model's episode or the
    # provider-mixed summary -- only user turns remain.
    hist = [_m("you", "act on it")]
    cross_ep = Episode(
        id="e1", timestamp=1.0, provider="codex", objective="o",
        actions=(), outcome="CROSS_EPISODE_OUTCOME", artifacts=(),
        tokens_consumed=0, raw_turn_count=1, source="live",
    )
    ctx = build_lane_context(
        hist, "openrouter", policy="off",
        episodes=[cross_ep], compaction_summary="MIXED_SUMMARY",
    )
    assert "CROSS_EPISODE_OUTCOME" not in ctx
    assert "MIXED_SUMMARY" not in ctx
    assert "act on it" in ctx


def test_off_policy_recovery_allows_same_provider_episode():
    hist = [_m("you", "act on it")]
    same_ep = Episode(
        id="e2", timestamp=1.0, provider="openrouter", objective="o",
        actions=(), outcome="SAME_EPISODE_OUTCOME", artifacts=(),
        tokens_consumed=0, raw_turn_count=1, source="live",
    )
    ctx = build_lane_context(
        hist, "openrouter", policy="off", episodes=[same_ep],
    )
    assert "SAME_EPISODE_OUTCOME" in ctx


def test_recovery_excerpt_stays_within_budget():
    hist = [_m("you", "act on it")]
    ep = Episode(
        id="e1", timestamp=1.0, provider="codex", objective="o",
        actions=(), outcome="HEAD_R " + ("z" * 20000) + " TAIL_R",
        artifacts=(), tokens_consumed=0, raw_turn_count=1, source="live",
    )
    ctx = build_lane_context(hist, "openrouter", episodes=[ep], max_chars=6000)
    assert len(ctx) <= 6000 + 400  # wrapper header allowance
    assert "[...]" in ctx


# --------------------------------------------------------------------------- #
# (K) sticky cross turns are not double-emitted, and are clipped
# --------------------------------------------------------------------------- #

def test_sticky_cross_episode_is_not_double_emitted():
    # Three cross turns: the latest two ride verbatim (sticky); the oldest
    # relies on its episode. The sticky turns' episodes must NOT also inject.
    messages = _msgs(
        ("you", "q1"),
        ("gemini", "OLDER content marker_OLD"),   # non-sticky -> episode only
        ("openai", "MID content marker_MID"),      # sticky -> verbatim only
        ("claude", "RECENT content marker_REC"),   # sticky -> verbatim only
        ("you", "act now, openrouter"),
    )
    episodes = [
        generate_episode("q1", "OLDER content marker_OLD", "gemini"),
        generate_episode("prev", "MID content marker_MID", "openai"),
        generate_episode("prev", "RECENT content marker_REC", "claude"),
    ]
    result = state_messages_to_provider(
        messages, "openrouter", policy="summary", episodes=episodes,
    )
    blob = "\n".join(m["content"] for m in result)
    # Sticky turns appear exactly once (verbatim), NOT also as episode context.
    assert blob.count("marker_MID") == 1
    assert blob.count("marker_REC") == 1
    assert "[Response from openai]" in blob
    # The non-sticky older cross turn is carried by its (still-injected) episode.
    assert "marker_OLD" in blob
    assert "[Prior session context]" in blob


def test_sticky_cross_content_is_clipped():
    huge = "HEAD_CROSS " + ("q" * (_STICKY_CROSS_MAX_CHARS * 2)) + " TAIL_CROSS"
    messages = _msgs(
        ("you", "look at codex output"),
        ("openai", huge),
        ("you", "now act, openrouter"),
    )
    result = state_messages_to_provider(messages, "openrouter", policy="summary")
    cross = next(m for m in result if m["content"].startswith("[Response from openai]"))
    # Clipped: head + tail survive, middle elided, overall bounded.
    assert "HEAD_CROSS" in cross["content"]
    assert "TAIL_CROSS" in cross["content"]
    assert "[...]" in cross["content"]
    assert len(cross["content"]) <= _STICKY_CROSS_MAX_CHARS + 100


def test_non_sticky_cross_episode_still_injects():
    # A single older cross turn beyond the sticky window keeps injecting.
    messages = _msgs(
        ("you", "q1"),
        ("gemini", "SOLO_OLD marker"),
        ("openai", "s1"),
        ("claude", "s2"),
        ("you", "go"),
    )
    episodes = [generate_episode("q1", "SOLO_OLD marker", "gemini")]
    result = state_messages_to_provider(
        messages, "openrouter", policy="summary", episodes=episodes,
    )
    blob = "\n".join(m["content"] for m in result)
    assert "SOLO_OLD marker" in blob
    assert "[Prior session context]" in blob


# --------------------------------------------------------------------------- #
# (N) char-budget trim removes synthetic (user, "Noted.") pairs atomically
# --------------------------------------------------------------------------- #

def test_budget_trim_never_leaves_orphan_noted_ack():
    # Two orchestration notices become (user, "Noted.") pairs. A tight budget
    # that pops the oldest user half must drop its ack too -- the payload must
    # never open on a dangling assistant "Noted.".
    messages = _msgs(
        ("system", "[Solve] first objective " + "A" * 400),
        ("system", "[Pipeline] second objective " + "B" * 400),
        ("you", "final ask " + "C" * 400),
    )
    result = state_messages_to_provider(
        messages, "openrouter", policy="summary", max_chars=900,
    )
    assert result, "expected a non-empty trimmed payload"
    head = result[0]
    assert not (head["role"] == "assistant" and head["content"] == "Noted.")
    # No stranded ack anywhere: every "Noted." is preceded by a user message.
    for i, m in enumerate(result):
        if m["role"] == "assistant" and m["content"] == "Noted.":
            assert i > 0 and result[i - 1]["role"] == "user"


def test_budget_trim_drops_cross_pair_atomically():
    # A verbatim cross turn is also a (user, "Noted.") pair; trimming it must
    # not strand the ack.
    messages = _msgs(
        ("openai", "cross handoff " + "X" * 500),
        ("you", "act on it " + "Y" * 500),
    )
    result = state_messages_to_provider(
        messages, "openrouter", policy="summary", max_chars=700,
    )
    for i, m in enumerate(result):
        if m["role"] == "assistant" and m["content"] == "Noted.":
            assert i > 0 and result[i - 1]["role"] == "user"
    if result:
        assert not (
            result[0]["role"] == "assistant" and result[0]["content"] == "Noted."
        )
