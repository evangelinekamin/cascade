"""Temporal-awareness markers in the provider payload (deterministic).

Messages carry immutable creation timestamps; state_messages_to_provider,
with_timeline=True, prepends bracketed time markers where time meaningfully
jumps -- first message, day changes, gaps >= 10 min, and always the latest
turn -- so the model knows when time passes without thrashing the cache.
"""

import time

from cascade.state import ChatMessage
from cascade.conversation import (
    TIMELINE_GAP_SECONDS,
    _humanize_gap,
    _timeline_marker,
    state_messages_to_provider,
)


def _msg(role, content, ts):
    return ChatMessage(role=role, content=content, timestamp=ts)


# A fixed local-noon epoch so strftime output is stable regardless of the day
# the test runs (avoids Date.now()-style nondeterminism).
def _at(day_offset=0, hour=12, minute=0):
    base = time.mktime((2026, 7, 21, hour, minute, 0, 0, 0, -1))
    return base + day_offset * 86400


class TestHumanizeGap:
    def test_minutes(self):
        assert _humanize_gap(15 * 60) == "15m later"

    def test_hours_and_minutes(self):
        assert _humanize_gap(2 * 3600 + 15 * 60) == "2h 15m later"

    def test_whole_hours(self):
        assert _humanize_gap(3 * 3600) == "3h later"

    def test_days(self):
        assert _humanize_gap(2 * 86400) == "2d later"


class TestTimelineMarker:
    def test_first_message_gets_full_datetime(self):
        m = _timeline_marker(_at(), prev_ts=0.0, force=False)
        assert m.startswith("[2026-07-21")
        assert "12:00" in m

    def test_small_gap_unmarked_unless_forced(self):
        prev = _at(hour=12, minute=0)
        now = _at(hour=12, minute=5)  # 5 min < threshold
        assert _timeline_marker(now, prev, force=False) == ""

    def test_small_gap_forced_shows_clock_only(self):
        prev = _at(hour=12, minute=0)
        now = _at(hour=12, minute=5)
        assert _timeline_marker(now, prev, force=True) == "[12:05]\n"

    def test_large_same_day_gap_shows_clock_and_relative(self):
        prev = _at(hour=12, minute=0)
        now = _at(hour=14, minute=15)
        assert _timeline_marker(now, prev, force=False) == "[14:15, 2h 15m later]\n"

    def test_day_change_shows_date(self):
        prev = _at(day_offset=0, hour=18)
        now = _at(day_offset=1, hour=9)
        m = _timeline_marker(now, prev, force=False)
        assert m.startswith("[2026-07-22")

    def test_zero_timestamp_never_marked(self):
        assert _timeline_marker(0.0, prev_ts=_at(), force=True) == ""

    def test_threshold_boundary_is_marked(self):
        prev = _at(hour=12, minute=0)
        now = prev + TIMELINE_GAP_SECONDS
        assert _timeline_marker(now, prev, force=False) != ""


class TestPayloadTimeline:
    def test_markers_injected_only_where_time_jumps(self):
        msgs = [
            _msg("you", "first", _at(hour=9, minute=0)),
            _msg("claude", "reply", _at(hour=9, minute=1)),   # small gap -> no marker
            _msg("you", "later", _at(hour=13, minute=0)),     # 3h59m -> marked
            _msg("claude", "reply2", _at(hour=13, minute=1)),
        ]
        out = state_messages_to_provider(msgs, "claude", with_timeline=True)
        assert out[0]["content"].startswith("[2026-07-21")     # first, full date
        assert out[1]["content"] == "reply"                    # small gap, unmarked
        assert out[2]["content"].startswith("[13:00,")         # big gap marked
        assert "first" in out[0]["content"]

    def test_last_turn_always_stamped(self):
        msgs = [
            _msg("you", "q1", _at(hour=9, minute=0)),
            _msg("claude", "a1", _at(hour=9, minute=1)),
            _msg("you", "q2", _at(hour=9, minute=2)),  # small gap but last -> forced
        ]
        out = state_messages_to_provider(msgs, "claude", with_timeline=True)
        assert out[-1]["content"] == "[09:02]\nq2"

    def test_no_timeline_by_default(self):
        msgs = [
            _msg("you", "q1", _at(hour=9)),
            _msg("claude", "a1", _at(hour=13)),
        ]
        out = state_messages_to_provider(msgs, "claude")  # with_timeline defaults False
        assert out[0]["content"] == "q1"
        assert out[1]["content"] == "a1"

    def test_gap_measured_between_visible_messages_under_summary(self):
        # A skipped cross-provider turn's elapsed time folds into the next
        # visible gap rather than resetting it.
        msgs = [
            _msg("you", "q1", _at(hour=9, minute=0)),
            _msg("gemini", "other-provider", _at(hour=9, minute=30)),  # skipped
            _msg("you", "q2", _at(hour=9, minute=45)),  # 45m since q1 (visible)
        ]
        out = state_messages_to_provider(msgs, "claude", policy="summary", with_timeline=True)
        # Only the two "you" turns survive; the second is marked vs the first.
        assert len(out) == 2
        assert out[1]["content"].startswith("[09:45, 45m later]")

    def test_full_policy_marks_cross_provider_turns(self):
        msgs = [
            _msg("you", "q1", _at(hour=9, minute=0)),
            _msg("gemini", "cross", _at(hour=12, minute=0)),  # 3h -> marked
        ]
        out = state_messages_to_provider(msgs, "claude", policy="full", with_timeline=True)
        # user q1, then [Response from gemini] (marked), then "Noted."
        assert out[1]["content"].startswith("[12:00, 3h later]\n[Response from gemini]")
