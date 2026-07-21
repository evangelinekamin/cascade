"""Autocomplete arrow-nav works even after prompt history exists (review fix)."""

import pytest
from textual.app import App, ComposeResult

from cascade.widgets.input_frame import InputFrame, ChatTextArea
from cascade.widgets.autocomplete import AutocompleteDropdown


class _Harness(App):
    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[str] = []

    def compose(self) -> ComposeResult:
        yield InputFrame(active_provider="claude", mode="plan")

    def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        self.submitted.append(event.value)


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
async def test_enter_accepts_completion_when_dropdown_open():
    app = _Harness()
    async with app.run_test() as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        dd = app.query_one(AutocompleteDropdown)
        ta.focus()
        await pilot.pause()

        # Partial command with the dropdown open on a highlighted match.
        from cascade.commands import get_matching_commands
        matches = get_matching_commands("m")
        assert matches, "expected at least one /m* command"
        ta.load_text("/m")
        dd.show(matches)
        await pilot.pause()
        assert dd.visible

        await pilot.press("enter")
        await pilot.pause()

        # Enter filled the completion instead of submitting a partial command.
        assert ta.text == f"/{matches[0].name} ", ta.text
        assert not dd.visible
        assert app.submitted == [], "Enter must not submit while the dropdown is open"


@pytest.mark.asyncio
async def test_enter_submits_when_dropdown_closed():
    app = _Harness()
    async with app.run_test() as pilot:
        ta = app.query_one("#main_input", ChatTextArea)
        dd = app.query_one(AutocompleteDropdown)
        ta.focus()
        await pilot.pause()

        ta.load_text("just a message")
        dd.hide()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.submitted == ["just a message"]


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
