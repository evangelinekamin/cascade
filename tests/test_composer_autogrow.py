"""The composer must auto-grow with its content (dogfood: "textbox does not expand").

Textual's TextArea is fixed-height and scrolls internally, which is what made
the composer "jump lines at random". These tests mount the widget with the
PRODUCTION stylesheet -- cascade.tcss pins `.main-input { height: 1 }`, so a
harness without it cannot reproduce the bug at all.
"""

from pathlib import Path

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Static

import cascade
from cascade.app import CascadeTUI
from cascade.history import HistoryDB
from cascade.widgets.input_frame import InputFrame, ChatTextArea, FramedInput

_TCSS = str(Path(cascade.__file__).parent / "cascade.tcss")


class _Harness(App):
    CSS_PATH = _TCSS

    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[str] = []

    def compose(self) -> ComposeResult:
        yield InputFrame(active_provider="claude", mode="plan")

    def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        self.submitted.append(event.value)


class _ScreenHarness(_Harness):
    """MainScreen's arrangement: a 1fr transcript above the docked composer."""

    CSS = "#transcript { height: 1fr; width: 100%; }"

    def compose(self) -> ComposeResult:
        yield Static("transcript", id="transcript")
        yield InputFrame(active_provider="claude", mode="plan")


async def _settle(pilot) -> None:
    """Drain edit -> layout -> resize -> rewrap before measuring."""
    for _ in range(3):
        await pilot.pause()


def _rendered_rows(ta: ChatTextArea) -> list[str]:
    """The composer's visible rows, actually rendered through render_line."""
    return [ta.render_line(y).text.rstrip() for y in range(ta.size.height)]


@pytest.mark.asyncio
async def test_starts_at_one_row():
    app = _Harness()
    async with app.run_test(size=(60, 24)) as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        ta.focus()
        await _settle(pilot)
        assert ta.size.height == 1


@pytest.mark.asyncio
async def test_grows_with_explicit_newlines():
    app = _Harness()
    async with app.run_test(size=(60, 24)) as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        frame = app.query_one(FramedInput)
        ta.focus()
        await _settle(pilot)
        before = frame.size.height

        ta.load_text("one\ntwo\nthree\nfour")
        await _settle(pilot)

        assert ta.size.height == 4
        # The bordered frame grows with it, otherwise the rows are invisible.
        assert frame.size.height == before + 3
        assert _rendered_rows(ta) == ["one", "two", "three", "four"]


@pytest.mark.asyncio
async def test_grows_with_soft_wrapped_long_line():
    """A single line with no newlines still occupies several rows."""
    app = _Harness()
    async with app.run_test(size=(60, 24)) as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        ta.focus()
        await _settle(pilot)

        ta.load_text("word " * 40)
        await _settle(pilot)

        assert "\n" not in ta.text
        assert ta.size.height > 1
        assert ta.size.height == ta.wrapped_document.height
        assert len(_rendered_rows(ta)) == ta.size.height


@pytest.mark.asyncio
async def test_capped_at_max_rows_and_scrolls_internally():
    app = _Harness()
    async with app.run_test(size=(60, 40)) as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        ta.focus()
        await _settle(pilot)

        ta.load_text("\n".join(f"line {i}" for i in range(40)))
        await _settle(pilot)

        assert ChatTextArea.MAX_ROWS == 12
        assert ta.size.height == ChatTextArea.MAX_ROWS
        # Past the cap the composer keeps its own scrollback.
        assert ta.virtual_size.height == 40
        assert len(_rendered_rows(ta)) == ChatTextArea.MAX_ROWS


@pytest.mark.asyncio
async def test_shrinks_back_when_content_is_deleted():
    app = _Harness()
    async with app.run_test(size=(60, 24)) as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        ta.focus()
        await _settle(pilot)

        ta.load_text("a\nb\nc\nd\ne")
        await _settle(pilot)
        assert ta.size.height == 5

        ta.load_text("a")
        await _settle(pilot)
        assert ta.size.height == 1


@pytest.mark.asyncio
async def test_rewraps_and_regrows_on_terminal_resize():
    app = _Harness()
    async with app.run_test(size=(80, 24)) as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        ta.focus()
        await _settle(pilot)

        ta.load_text("word " * 20)
        await _settle(pilot)
        wide = ta.size.height
        assert wide > 1

        await pilot.resize_terminal(40, 24)
        await _settle(pilot)
        narrow = ta.size.height

        assert narrow > wide
        assert narrow == ta.wrapped_document.height


@pytest.mark.asyncio
async def test_growth_squeezes_the_transcript_without_clipping_the_composer():
    app = _ScreenHarness()
    async with app.run_test(size=(60, 24)) as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        transcript = app.query_one("#transcript", Static)
        ta.focus()
        await _settle(pilot)
        tall_transcript = transcript.size.height

        ta.load_text("\n".join(f"line {i}" for i in range(40)))
        await _settle(pilot)

        assert transcript.size.height == tall_transcript - (ChatTextArea.MAX_ROWS - 1)
        # The whole composer stays on screen -- growth must not run off the bottom.
        region = app.query_one(FramedInput).region
        assert region.y >= 0 and region.bottom <= 24


@pytest.mark.asyncio
async def test_multiline_paste_stays_editable_and_grows():
    app = _Harness()
    async with app.run_test(size=(60, 24)) as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        ta.focus()
        await _settle(pilot)

        # Delivered the way the driver does it: to the app, then forwarded.
        app.post_message(events.Paste("first\nsecond\nthird"))
        await _settle(pilot)

        assert ta.text == "first\nsecond\nthird"
        assert ta.size.height == 3
        # Still a live document, not an opaque placeholder.
        await pilot.press("!")
        await _settle(pilot)
        assert ta.text == "first\nsecond\nthird!"


@pytest.mark.asyncio
async def test_short_terminal_caps_below_max_rows():
    """A small pane must not let the composer push its own frame off-screen."""
    app = _ScreenHarness()
    async with app.run_test(size=(60, 10)) as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        ta.focus()
        await _settle(pilot)

        ta.load_text("\n".join(f"line {i}" for i in range(40)))
        await _settle(pilot)

        assert ta.size.height == 5  # half of a 10-row terminal, not MAX_ROWS
        region = app.query_one(FramedInput).region
        assert region.y >= 0 and region.bottom <= 10


@pytest.mark.asyncio
async def test_grows_inside_the_real_app(tmp_path, monkeypatch):
    """End to end: the actual CascadeTUI screen, stylesheet and dock order."""
    monkeypatch.setattr(
        "cascade.app.HistoryDB", lambda: HistoryDB(db_path=str(tmp_path / "h.db"))
    )
    app = CascadeTUI(cli_app=None)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(10):  # MainScreen is pushed from CascadeTUI.on_mount
                await pilot.pause()
            ta = app.screen.query_one("#main_input", ChatTextArea)
            frame = app.screen.query_one(FramedInput)
            assert ta.size.height == 1

            ta.load_text("a\nb\nc\nd\ne\nf")
            await _settle(pilot)
            assert ta.size.height == 6

            ta.load_text("\n".join(str(i) for i in range(50)))
            await _settle(pilot)
            assert ta.size.height == ChatTextArea.MAX_ROWS
            assert frame.region.y >= 0 and frame.region.bottom <= 30
            assert _rendered_rows(ta) == [str(i) for i in range(12)]
    finally:
        app.db.close()


@pytest.mark.asyncio
async def test_typing_a_newline_grows_then_enter_submits():
    """Real key path: ctrl+j keeps composing, Enter still submits everything."""
    app = _Harness()
    async with app.run_test(size=(60, 24)) as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        ta.focus()
        await _settle(pilot)

        await pilot.press("h", "i")
        await pilot.press("ctrl+j")
        await pilot.press("y", "o")
        await _settle(pilot)

        assert ta.text == "hi\nyo"
        assert ta.size.height == 2

        await pilot.press("enter")
        await _settle(pilot)
        assert app.submitted == ["hi\nyo"]
        # Submission is the caller's job to clear; the widget itself keeps its size.
        assert ta.size.height == 2
