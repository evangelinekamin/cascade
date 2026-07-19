"""Floating bordered input with provider accent border.

Rounded border in accent color, provider name as border-title,
token count as border-subtitle. Interior: prompt char > + Input.
Below: mode indicator with shift+tab hint.
Autocomplete dropdown appears when typing slash commands.
"""

from rich.text import Text
from textual import events
from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Input, Static, Label
from textual.reactive import reactive

from ..theme import PALETTE, MODES, get_accent
from ..commands import get_matching_commands
from .autocomplete import AutocompleteDropdown


class ChatInput(Input):
    """Input with multiline paste capture and prompt history (up/down arrow).

    Multiline paste: stores full text, shows ``[pasted N chars]``.
    History: up/down arrows navigate previous submissions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_paste: str | None = None
        self._history: list[str] = []
        self._history_idx: int = -1
        self._draft: str = ""

    def record(self, text: str) -> None:
        """Record a submitted prompt into history."""
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._history_idx = -1
        self._draft = ""

    def _on_paste(self, event: events.Paste) -> None:
        if not event.text:
            event.stop()
            return
        if "\n" in event.text:
            self._pending_paste = event.text.strip()
            n = len(self._pending_paste)
            self.value = f"[pasted {n} chars]"
            self.cursor_position = len(self.value)
        else:
            self._pending_paste = None
            # Set value directly to avoid double-insertion
            pos = self.cursor_position
            self.value = self.value[:pos] + event.text + self.value[pos:]
            self.cursor_position = pos + len(event.text)
        event.stop()
        event.prevent_default()

    async def _on_key(self, event: events.Key) -> None:
        # Up/down arrow for prompt history navigation
        if event.key == "up" and self._history:
            if self._history_idx == -1:
                self._draft = self.value
                self._history_idx = len(self._history) - 1
            elif self._history_idx > 0:
                self._history_idx -= 1
            else:
                return
            self.value = self._history[self._history_idx]
            self.cursor_position = len(self.value)
            event.stop()
            event.prevent_default()
        elif event.key == "down" and self._history_idx >= 0:
            if self._history_idx < len(self._history) - 1:
                self._history_idx += 1
                self.value = self._history[self._history_idx]
            else:
                self._history_idx = -1
                self.value = self._draft
            self.cursor_position = len(self.value)
            event.stop()
            event.prevent_default()
        else:
            await super()._on_key(event)


class InputFrame(Widget):
    """The bottom input region: framed text area + mode indicator."""

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

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update autocomplete suggestions as user types."""
        value = event.value
        dropdown = self.query_one(AutocompleteDropdown)

        if value.startswith("/") and len(value) > 0:
            prefix = value[1:]  # strip the /
            matches = get_matching_commands(prefix)
            if matches and value != f"/{matches[0].name}":
                dropdown.show(matches)
            else:
                dropdown.hide()
        else:
            dropdown.hide()

    def on_key(self, event) -> None:
        """Handle arrow keys and tab for autocomplete navigation."""
        dropdown = self.query_one(AutocompleteDropdown)
        if not dropdown.visible:
            return

        if event.key == "down":
            dropdown.move_selection(1)
            event.prevent_default()
            event.stop()
        elif event.key == "up":
            dropdown.move_selection(-1)
            event.prevent_default()
            event.stop()
        elif event.key == "tab":
            selected = dropdown.selected_command
            if selected:
                try:
                    inp = self.query_one("#main_input", Input)
                    inp.value = f"/{selected} "
                    inp.cursor_position = len(inp.value)
                except Exception:
                    pass
                dropdown.hide()
            event.prevent_default()
            event.stop()
        elif event.key == "escape":
            dropdown.hide()
            event.prevent_default()
            event.stop()


class FramedInput(Widget):
    """The bordered container holding the prompt char and Input widget."""

    DEFAULT_CSS = """
    FramedInput {
        height: 3;
        width: 100%;
        border: solid #b44dff;
        background: #0d1117;
        padding: 0 1;
        layout: horizontal;
    }
    """

    def __init__(self, provider: str, context_label: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._provider = provider
        self._context_label = context_label

    def compose(self) -> ComposeResult:
        yield Label("\u276f", id="prompt_char", classes="prompt-char")
        yield ChatInput(placeholder="", id="main_input", classes="main-input")

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
        t.append("\u2500\u2500\u2500 ", style=accent)
        t.append(self._mode, style=f"bold {accent}")
        t.append(" \u2500\u2500\u2500 ", style=accent)
        t.append("shift+tab", style=f"dim {PALETTE.text_dim}")
        return t
