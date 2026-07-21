"""Autocomplete arrow-nav works even after prompt history exists (review fix)."""

import pytest
from textual.app import App, ComposeResult

from cascade.widgets.input_frame import InputFrame, ChatTextArea
from cascade.widgets.autocomplete import AutocompleteDropdown


class _Harness(App):
    def compose(self) -> ComposeResult:
        yield InputFrame(active_provider="claude", mode="plan")


@pytest.mark.asyncio
async def test_up_nav_moves_dropdown_not_history_when_open():
    app = _Harness()
    async with app.run_test() as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        dd = app.query_one(AutocompleteDropdown)
        ta.focus()
        await pilot.pause()
        # Populate history (the condition that used to break up-nav).
        ta.record("an earlier prompt")
        # Open the autocomplete on a slash command.
        ta.load_text("/")
        await pilot.pause()
        # Force the dropdown visible with >1 suggestion.
        from cascade.commands import get_matching_commands
        dd.show(get_matching_commands(""))
        await pilot.pause()
        assert dd.visible and dd._selected_idx == 0
        # Press Up: dropdown selection moves (NOT history recall).
        await pilot.press("up")
        await pilot.pause()
        assert dd._selected_idx != 0, "up should move the dropdown selection"
        # The composer text was NOT replaced by a history entry.
        assert ta.text == "/", ta.text


@pytest.mark.asyncio
async def test_up_recalls_history_when_dropdown_closed():
    app = _Harness()
    async with app.run_test() as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        ta.focus()
        await pilot.pause()
        ta.record("earlier prompt")
        ta.load_text("")
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        assert ta.text == "earlier prompt"
