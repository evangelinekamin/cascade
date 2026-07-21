"""Floating bordered input with provider accent border.

Rounded border in accent color, provider name as border-title, context
occupancy as border-subtitle. Interior: prompt char > + a multiline
composer. Below: mode indicator with shift+tab hint. Autocomplete
dropdown appears when typing slash commands.
"""

from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Static, Label, TextArea
from textual.reactive import reactive

from ..theme import PALETTE, MODES, get_accent
from ..commands import get_matching_commands
from .autocomplete import AutocompleteDropdown


class ChatTextArea(TextArea):
    """Auto-growing multiline prompt composer.

    - Enter submits (posts ``ChatTextArea.Submitted``).
    - shift+enter (modern terminals) or ctrl+j inserts a newline.
    - Pasted text is inserted verbatim and stays fully editable -- no
      opaque ``[pasted N chars]`` placeholder.
    - Up/Down navigate prompt history only when the cursor is on the
      first/last line; otherwise they move between lines as normal.
    """

    class Submitted(Message):
        """The user submitted the composed prompt."""

        def __init__(self, widget: "ChatTextArea", value: str) -> None:
            super().__init__()
            self.text_area = widget
            self.value = value

    BINDINGS = [
        Binding("shift+enter", "newline", "Newline", show=False),
        Binding("ctrl+j", "newline", "Newline", show=False),
    ]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("soft_wrap", True)
        kwargs.setdefault("show_line_numbers", False)
        kwargs.setdefault("tab_behavior", "focus")  # Tab leaves for autocomplete
        super().__init__(**kwargs)
        self._history: list[str] = []
        self._history_idx: int = -1
        self._draft: str = ""

    # -- history --------------------------------------------------------

    def record(self, text: str) -> None:
        """Record a submitted prompt into history."""
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._history_idx = -1
        self._draft = ""

    # -- value compatibility (callers used the Input API) ---------------

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, v: str) -> None:
        self.load_text(v or "")

    def clear_value(self) -> None:
        self.load_text("")

    # -- actions --------------------------------------------------------

    def action_newline(self) -> None:
        self.insert("\n")

    def _submit(self) -> None:
        self.post_message(self.Submitted(self, self.text))

    def _cursor_row(self) -> int:
        loc = self.cursor_location
        return loc[0] if loc else 0

    def _last_row(self) -> int:
        try:
            return self.document.line_count - 1
        except Exception:
            return 0

    def _autocomplete(self):
        """The visible autocomplete dropdown, if one is open."""
        try:
            dd = self.screen.query_one(AutocompleteDropdown)
            return dd if dd.visible else None
        except Exception:
            return None

    async def _on_key(self, event: events.Key) -> None:
        key = event.key

        # When the slash-command autocomplete is open, arrows/tab/escape drive
        # it -- handled HERE (the composer receives keys first, so relying on
        # the InputFrame ancestor lost the race to history recall).
        dropdown = self._autocomplete()
        if dropdown is not None and key in ("up", "down", "tab", "escape"):
            if key == "down":
                dropdown.move_selection(1)
            elif key == "up":
                dropdown.move_selection(-1)
            elif key == "tab":
                selected = dropdown.selected_command
                if selected:
                    self.load_text(f"/{selected} ")
                    self.move_cursor(self.document.end)
                dropdown.hide()
            else:  # escape
                dropdown.hide()
            event.stop()
            event.prevent_default()
            return

        # Enter submits (shift+enter / ctrl+j newline handled by BINDINGS).
        if key == "enter":
            event.stop()
            event.prevent_default()
            self._submit()
            return

        # History recall only at the first/last line, so multi-line editing
        # still uses Up/Down to move between lines.
        if key == "up" and self._history and self._cursor_row() == 0:
            if self._history_idx == -1:
                self._draft = self.text
                self._history_idx = len(self._history) - 1
            elif self._history_idx > 0:
                self._history_idx -= 1
            else:
                event.stop()
                event.prevent_default()
                return
            self.load_text(self._history[self._history_idx])
            self.move_cursor(self.document.end)
            event.stop()
            event.prevent_default()
            return
        if key == "down" and self._history_idx >= 0 and self._cursor_row() == self._last_row():
            if self._history_idx < len(self._history) - 1:
                self._history_idx += 1
                self.load_text(self._history[self._history_idx])
            else:
                self._history_idx = -1
                self.load_text(self._draft)
            self.move_cursor(self.document.end)
            event.stop()
            event.prevent_default()
            return

        await super()._on_key(event)


class InputFrame(Widget):
    """The bottom input region: framed composer + mode indicator."""

    DEFAULT_CSS = """
    InputFrame {
        height: auto;
        width: 100%;
        dock: bottom;
        padding: 0 2 1 2;
    }
    """

    active_provider: reactive[str] = reactive("gemini")
    mode: reactive[str] = reactive("design")
    # Context-occupancy label for the border subtitle (e.g. "ctx 12.4k · 7%").
    # Empty hides the subtitle. This is window fill, NOT cumulative spend --
    # the session total lives on the exit screen.
    context_label: reactive[str] = reactive("")

    def __init__(
        self,
        active_provider: str = "gemini",
        mode: str = "design",
        context_label: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.active_provider = active_provider
        self.mode = mode
        self.context_label = context_label

    def compose(self) -> ComposeResult:
        yield AutocompleteDropdown(id="autocomplete")
        yield FramedInput(self.active_provider, self.context_label)
        yield ModeIndicator(self.mode)

    def watch_active_provider(self, value: str) -> None:
        try:
            self.query_one(FramedInput).set_provider(value)
        except Exception:
            pass

    def watch_mode(self, value: str) -> None:
        try:
            self.query_one(ModeIndicator).set_mode(value)
        except Exception:
            pass

    def watch_context_label(self, value: str) -> None:
        try:
            self.query_one(FramedInput).set_context_label(value)
        except Exception:
            pass

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Update autocomplete suggestions as the user types a slash command."""
        value = event.text_area.text
        dropdown = self.query_one(AutocompleteDropdown)
        # Only a single-line command line triggers autocomplete.
        if value.startswith("/") and "\n" not in value:
            matches = get_matching_commands(value[1:])
            if matches and value != f"/{matches[0].name}":
                dropdown.show(matches)
            else:
                dropdown.hide()
        else:
            dropdown.hide()

    # Autocomplete arrow/tab/escape navigation is handled by ChatTextArea
    # itself (it receives keys first as the focused widget), so no ancestor
    # on_key is needed here.


class FramedInput(Widget):
    """The bordered container holding the prompt char and the composer."""

    DEFAULT_CSS = """
    FramedInput {
        height: auto;
        min-height: 3;
        width: 100%;
        border: solid #b44dff;
        background: #0d1117;
        padding: 0 1;
        layout: horizontal;
    }
    FramedInput #main_input {
        height: auto;
        min-height: 1;
        max-height: 12;
        width: 1fr;
        background: #0d1117;
        border: none;
        padding: 0;
    }
    FramedInput #main_input:focus {
        border: none;
    }
    """

    def __init__(self, provider: str, context_label: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._provider = provider
        self._context_label = context_label

    def compose(self) -> ComposeResult:
        yield Label("❯", id="prompt_char", classes="prompt-char")
        yield ChatTextArea(id="main_input", classes="main-input")

    def on_mount(self) -> None:
        self._apply_accent()

    def set_provider(self, provider: str) -> None:
        self._provider = provider
        self._apply_accent()

    def set_context_label(self, label: str) -> None:
        self._context_label = label
        self._apply_accent()

    def _apply_accent(self) -> None:
        accent = get_accent(self._provider)
        self.styles.border = ("solid", accent)

        self.border_title = Text(f" {self._provider} ", style=f"bold {accent}")

        if self._context_label:
            self.border_subtitle = Text(
                f" {self._context_label} ", style=f"dim {PALETTE.text_dim}",
            )
        else:
            self.border_subtitle = None

        try:
            prompt = self.query_one("#prompt_char")
            prompt.styles.color = accent
            prompt.styles.text_style = "bold"
        except Exception:
            pass


class ModeIndicator(Static):
    """Single line below the input frame showing current mode."""

    DEFAULT_CSS = """
    ModeIndicator {
        height: 1;
        width: 100%;
        text-align: center;
        padding: 0 0 0 2;
    }
    """

    def __init__(self, mode: str = "design", **kwargs) -> None:
        super().__init__(**kwargs)
        self._mode = mode

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.refresh()

    def render(self) -> Text:
        provider = MODES.get(self._mode, {"provider": "gemini"})["provider"]
        accent = get_accent(provider)
        t = Text()
        # Frame the mode name in its own accent (the dashes were double-dimmed to
        # near-black); the bold name still stands out against the plain-weight rule.
        t.append("─── ", style=accent)
        t.append(self._mode, style=f"bold {accent}")
        t.append(" ─── ", style=accent)
        t.append("shift+enter for newline · shift+tab mode", style=f"dim {PALETTE.text_dim}")
        return t
