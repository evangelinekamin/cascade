"""Composer auto-grow interactions with the real screen (review findings).

Two failure modes the synthetic two-widget harness could not see:
  1. The row cap is half the viewport, but the composer's height is pinned by
     an inline style -- so a HEIGHT-ONLY terminal resize changes the cap while
     delivering no Resize to the composer, and a tall composer keeps its old
     height and pushes its own top border off-screen.
  2. ChatHistory is height:1fr, so a growing composer shrinks it and raises
     max_scroll_y while scroll_offset stays put -- silently hiding the newest
     messages on the very surface the dogfood report was about.
"""

import pytest

from cascade.app import CascadeTUI
from cascade.history import HistoryDB
from cascade.screens.main import MainScreen
from cascade.widgets.input_frame import ChatTextArea
from cascade.widgets.message import ChatHistory, MessageWidget


@pytest.fixture
def app(tmp_path, monkeypatch):
    db_path = tmp_path / "history.db"
    monkeypatch.setattr("cascade.app.HistoryDB", lambda: HistoryDB(db_path=str(db_path)))
    application = CascadeTUI(cli_app=None)
    yield application
    application.db.close()


@pytest.mark.asyncio
async def test_height_only_resize_keeps_the_composer_on_screen(app):
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        composer = app.screen.query_one("#main_input", ChatTextArea)
        composer.load_text("\n".join(f"line {i}" for i in range(40)))
        await pilot.pause()
        assert composer.size.height == composer.MAX_ROWS  # capped at 12

        # Shrink HEIGHT only -- width unchanged, so the composer gets no Resize
        # of its own and only the screen-level hook can save it.
        await pilot.resize_terminal(100, 12)
        await pilot.pause()
        await pilot.pause()

        assert composer.size.height <= composer._row_cap()
        frame = composer.parent
        assert frame.region.y >= 0, "composer pushed its own top border off-screen"


@pytest.mark.asyncio
async def test_growing_the_composer_keeps_the_newest_message_visible(app):
    async with app.run_test(size=(100, 44)) as pilot:
        await pilot.pause()
        chat = app.screen.query_one(ChatHistory)
        for i in range(30):
            chat.mount(MessageWidget("you" if i % 2 else "claude", f"message {i}"))
        await pilot.pause()
        chat.scroll_end(animate=False)
        await pilot.pause()
        assert chat.scroll_offset.y >= chat.max_scroll_y - 1  # pinned at bottom

        composer = app.screen.query_one("#main_input", ChatTextArea)
        composer.load_text("a\nb\nc\nd\ne")
        await pilot.pause()
        await pilot.pause()

        # The transcript shrank, but the tail must still be on screen.
        assert chat.scroll_offset.y >= chat.max_scroll_y - 1


@pytest.mark.asyncio
async def test_scrolled_back_user_is_not_yanked_to_the_bottom(app):
    async with app.run_test(size=(100, 44)) as pilot:
        await pilot.pause()
        chat = app.screen.query_one(ChatHistory)
        for i in range(40):
            chat.mount(MessageWidget("claude", f"message {i}"))
        await pilot.pause()
        chat.scroll_to(y=0, animate=False)  # deliberately reading history
        await pilot.pause()

        composer = app.screen.query_one("#main_input", ChatTextArea)
        composer.load_text("a\nb\nc\nd\ne\nf")
        await pilot.pause()
        await pilot.pause()

        assert chat.scroll_offset.y < 5, "a scrolled-back reader was yanked to the tail"


@pytest.mark.asyncio
async def test_composer_tracks_its_row_count(app):
    # Do NOT intercept post_message to observe this -- that widget method is
    # Textual's whole message pump, and stubbing it deadlocks the pilot.
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        composer = app.screen.query_one("#main_input", ChatTextArea)
        assert composer._rows == 1

        composer.load_text("a\nb\nc")
        await pilot.pause()
        assert composer._rows == 3

        composer.load_text("")
        await pilot.pause()
        assert composer._rows == 1
