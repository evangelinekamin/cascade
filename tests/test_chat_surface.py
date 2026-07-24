"""Bottom-anchored turn indicator + queued-prompt preview/retract.

Both widgets are RENDERED for real -- either mounted in the production
CascadeTUI (pilot + pause) or driven through a Rich Console -- because the
regressions these guard against (a bad style string, a widget that is
constructed but never laid out) only surface at render time.
"""

import io
import time
from pathlib import Path

import pytest
from rich.console import Console

import cascade
from cascade.app import CascadeTUI
from cascade.history import HistoryDB
from cascade.screens.main import MainScreen
from cascade.swarm.lifecycle import RunContext
from cascade.widgets.input_frame import InputFrame, ChatTextArea
from cascade.widgets.message import (
    QueuePreview,
    QueuedPromptRow,
    TurnIndicator,
    summarize_queued_prompt,
)

_TCSS = str(Path(cascade.__file__).parent / "cascade.tcss")


# ---------------------------------------------------------------------------
# Pure-render unit checks (no mount)
# ---------------------------------------------------------------------------

def _console_render(renderable) -> str:
    console = Console(width=72, file=io.StringIO(), no_color=True)
    console.print(renderable)
    return console.file.getvalue()


def test_summarize_queued_prompt_passthrough_truncate_and_line_count():
    assert summarize_queued_prompt("ship it") == "ship it"

    capped = summarize_queued_prompt("x" * 120)
    assert len(capped) <= 56 and capped.endswith("…")

    multi = summarize_queued_prompt("first line\nsecond\nthird")
    assert multi.startswith("first line")
    assert "(+2 lines)" in multi


def test_turn_indicator_render_shows_label_and_elapsed_clock():
    ti = TurnIndicator()
    ti._label = "calling read_file..."
    ti._started_at = time.monotonic() - 3
    out = _console_render(ti.render())
    assert "calling read_file..." in out
    assert "3s" in out
    # The spinner glyph is present (first frame while idle).
    assert TurnIndicator.SPINNER_FRAMES[0] in out


def test_queued_prompt_row_render_shows_summary_and_affordance():
    row = QueuedPromptRow(0, "add tests\nfor the parser")
    out = _console_render(row.render())
    assert "add tests" in out
    assert "queued" in out
    assert "×" in out  # retract affordance glyph


def test_retract_out_of_range_index_is_a_noop():
    screen = MainScreen()
    screen._queued_prompts.extend(["a", "b"])
    screen.on_queue_preview_retract_requested(QueuePreview.RetractRequested(9))
    assert list(screen._queued_prompts) == ["a", "b"]


# ---------------------------------------------------------------------------
# Rendered in the real app
# ---------------------------------------------------------------------------

async def _boot(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cascade.app.HistoryDB", lambda: HistoryDB(db_path=str(tmp_path / "h.db"))
    )
    app = CascadeTUI(cli_app=None)
    return app


async def _settle(pilot, n: int = 10) -> None:
    for _ in range(n):
        await pilot.pause()


@pytest.mark.asyncio
async def test_turn_indicator_pins_above_input_and_collapses_when_idle(
    tmp_path, monkeypatch
):
    app = await _boot(tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await _settle(pilot)
            screen = app.screen
            ti = screen.query_one(TurnIndicator)
            frame = screen.query_one(InputFrame)

            # Idle: collapsed to zero height, contributing nothing.
            assert not ti.active
            assert ti.region.height == 0

            ti.start("claude", "thinking...")
            await pilot.pause()
            assert ti.active
            assert ti.region.height == 1
            # Actually laid out and rendered (not just constructed).
            assert ti.render_line(0).text.strip() != ""
            # Docked at the bottom of the chat area, above the input frame.
            assert ti.region.y >= 0
            assert ti.region.bottom <= frame.region.y

            ti.stop()
            await pilot.pause()
            assert not ti.active
            assert ti.region.height == 0
    finally:
        app.db.close()


@pytest.mark.asyncio
async def test_queue_preview_appears_renders_and_retracts_by_click(
    tmp_path, monkeypatch
):
    app = await _boot(tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await _settle(pilot)
            screen = app.screen
            preview = screen.query_one(QueuePreview)

            # Empty queue -> collapsed.
            assert preview.region.height == 0

            # Force the app busy so submits queue instead of dispatching.
            screen._active_run = RunContext(objective="busy", workflow="chat")

            ta = screen.query_one("#main_input", ChatTextArea)
            ta.focus()
            ta.load_text("first queued task")
            await pilot.press("enter")
            await pilot.pause()
            ta.load_text("second queued task")
            await pilot.press("enter")
            await _settle(pilot, 4)

            # The existing FIFO queue is intact and mirrored into the preview.
            assert list(screen._queued_prompts) == [
                "first queued task",
                "second queued task",
            ]
            rows = list(screen.query(QueuedPromptRow))
            assert len(rows) == 2
            assert preview.region.height >= 2
            assert "first queued task" in rows[0].render_line(0).text

            # Retract the first prompt by clicking its row.
            await pilot.click(rows[0])
            await _settle(pilot, 4)

            assert list(screen._queued_prompts) == ["second queued task"]
            rows_after = list(screen.query(QueuedPromptRow))
            assert len(rows_after) == 1
            assert "second queued task" in rows_after[0].render_line(0).text

            # Retracting the last one collapses the preview again.
            await pilot.click(rows_after[0])
            await _settle(pilot, 4)
            assert list(screen._queued_prompts) == []
            assert screen.query_one(QueuePreview).region.height == 0
    finally:
        app.db.close()
