"""Scroll-back re-mount: trimmed messages are reachable again in the live UI.

``trim_overflow`` moves the oldest messages into ``_overflow`` and shows a
clickable banner. Clicking it (``restore_page``) re-mounts a page of the most
recently-trimmed messages, in chronological order, directly below the banner --
so a long agentic session's history is never stranded (previously it was only
reachable via ``/export``).
"""

import pytest
from textual.app import App, ComposeResult

from cascade.widgets.message import (
    ChatHistory,
    MessageWidget,
    OverflowIndicator,
)


class _Harness(App):
    def compose(self) -> ComposeResult:
        yield ChatHistory(max_widgets=5)


def _bodies(chat):
    """Content strings of the currently-mounted MessageWidgets, top-to-bottom."""
    return [w._content for w in chat.children if isinstance(w, MessageWidget)]


@pytest.mark.asyncio
async def test_restore_page_remounts_trimmed_messages_in_order():
    app = _Harness()
    async with app.run_test() as pilot:
        chat = app.query_one(ChatHistory)
        await pilot.pause()

        # 12 messages into a cap-5 history -> 7 get trimmed to overflow.
        for i in range(12):
            chat.mount(MessageWidget("you", f"m{i}"))
        await chat.trim_overflow()
        await pilot.pause()

        assert _bodies(chat) == [f"m{i}" for i in range(7, 12)]
        assert chat._overflow == [("you", f"m{i}") for i in range(7)]
        assert chat._overflow_indicator is not None

        # Restore a page (all 7 fit under the default page size).
        await chat.restore_page()
        await pilot.pause()

        # The full chronological order is back, oldest at the top.
        assert _bodies(chat) == [f"m{i}" for i in range(12)]
        # Overflow drained; banner removed when nothing remains.
        assert chat._overflow == []
        assert chat._overflow_indicator is None


@pytest.mark.asyncio
async def test_restore_page_pages_newest_first_and_keeps_banner():
    app = _Harness()
    async with app.run_test() as pilot:
        chat = app.query_one(ChatHistory)
        chat._RESTORE_PAGE = 3  # small page to force multiple clicks
        await pilot.pause()

        for i in range(12):
            chat.mount(MessageWidget("you", f"m{i}"))
        await chat.trim_overflow()
        await pilot.pause()
        # overflow = m0..m6 (7 msgs), shown = m7..m11
        assert chat._overflow == [("you", f"m{i}") for i in range(7)]

        # First restore pulls the NEWEST-trimmed page (m4,m5,m6) adjacent to
        # what's shown, so the reconstructed window stays contiguous.
        await chat.restore_page()
        await pilot.pause()
        assert _bodies(chat) == [f"m{i}" for i in range(4, 12)]
        assert chat._overflow == [("you", f"m{i}") for i in range(4)]
        assert chat._overflow_indicator is not None  # banner persists

        # Second restore pulls the next page back (m1,m2,m3).
        await chat.restore_page()
        await pilot.pause()
        assert _bodies(chat) == [f"m{i}" for i in range(1, 12)]
        assert chat._overflow == [("you", "m0")]


@pytest.mark.asyncio
async def test_clicking_banner_triggers_restore():
    app = _Harness()
    async with app.run_test() as pilot:
        chat = app.query_one(ChatHistory)
        await pilot.pause()
        for i in range(12):
            chat.mount(MessageWidget("you", f"m{i}"))
        await chat.trim_overflow()
        await pilot.pause()

        indicator = chat.query_one(OverflowIndicator)
        indicator.post_message(OverflowIndicator.RestoreRequested())
        await pilot.pause()

        assert _bodies(chat) == [f"m{i}" for i in range(12)]
