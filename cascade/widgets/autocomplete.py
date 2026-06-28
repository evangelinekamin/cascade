"""Slash command autocomplete dropdown.

Shows matching commands as the user types `/` followed by characters.
Rendered as a floating overlay above the input frame.
"""

from rich.text import Text
from textual.widget import Widget
from textual.widgets import Static

from ..theme import PALETTE
from ..commands import CommandDef


class CommandSuggestion(Static):
    """A single suggestion row in the autocomplete dropdown."""

    DEFAULT_CSS = """
    CommandSuggestion {
        height: 1;
        width: 100%;
        padding: 0 1;
    }
    CommandSuggestion.selected {
        background: #1c2128;
    }
    """

    def __init__(self, cmd: CommandDef, selected: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cmd = cmd
        if selected:
            self.add_class("selected")

    def render(self) -> Text:
        t = Text()
        if self.has_class("selected"):
            t.append("\u258c", style=f"bold {PALETTE.cyan}")
        else:
            t.append(" ")
        t.append(f"/{self._cmd.name}", style=f"bold {PALETTE.text_bright}")
        t.append("  ", style="")
        t.append(self._cmd.description, style=f"dim {PALETTE.text_muted}")
        return t


class AutocompleteDropdown(Widget):
    """Floating dropdown that shows matching slash commands.

    Mount this inside the InputFrame. Call update_suggestions() on each
    keystroke. Set visible via display property.
    """

    DEFAULT_CSS = """
    AutocompleteDropdown {
        height: auto;
        max-height: 10;
        width: 100%;
        background: #161b22;
        border: solid #30363d;
        padding: 0;
        display: none;
        layer: overlay;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._suggestions: list[CommandDef] = []
        self._selected_idx: int = 0

    def show(self, suggestions: list[CommandDef]) -> None:
        """Show the dropdown with the given suggestions."""
        self._suggestions = suggestions
        self._selected_idx = 0
        if suggestions:
            self.display = True
        else:
            self.display = False
        self._rebuild()

    def hide(self) -> None:
        """Hide the dropdown."""
        self.display = False
        self._suggestions = []
        self._selected_idx = 0

    @property
    def visible(self) -> bool:
        return self.display is True or self.display is None

    @property
    def selected_command(self) -> str | None:
        """Return the currently selected command name, or None."""
        if self._suggestions and 0 <= self._selected_idx < len(self._suggestions):
            return self._suggestions[self._selected_idx].name
        return None

    def move_selection(self, delta: int) -> None:
        """Move the selected index up/down."""
        if not self._suggestions:
            return
        self._selected_idx = (self._selected_idx + delta) % len(self._suggestions)
        self._rebuild()

    def _rebuild(self) -> None:
        """Rebuild the suggestion widgets."""
        self.remove_children()
        for i, cmd in enumerate(self._suggestions):
            row = CommandSuggestion(cmd, selected=(i == self._selected_idx))
            self.mount(row)
