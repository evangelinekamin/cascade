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
    token_count: reactive[int] = reactive(0)

    def __init__(
        self,
        active_provider: str = "gemini",
        mode: str = "design",
        token_count: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.active_provider = active_provider
        self.mode = mode
        self.token_count = token_count

    def compose(self) -> ComposeResult:
        yield AutocompleteDropdown(id="autocomplete")
        yield FramedInput(self.active_provider, self.token_count)
        yield ModeIndicator(self.mode)

    def watch_active_provider(self, value: str) -> None:
        try:
            self.query_one(FramedInput).set_provider(value, self.token_count)
        except Exception:
            pass

    def watch_mode(self, value: str) -> None:
        try:
            self.query_one(ModeIndicator).set_mode(value)
        except Exception:
            pass

    def watch_token_count(self, value: int) -> None:
        try:
            self.query_one(FramedInput).set_provider(self.active_provider, value)
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
        border: round #b44dff;
        background: #121218;
        padding: 0 1;
        layout: horizontal;
    }
    """

    def __init__(self, provider: str, token_count: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._provider = provider
        self._token_count = token_count

    def compose(self) -> ComposeResult:
        yield Label("\u276f", id="prompt_char", classes="prompt-char")
        yield ChatInput(placeholder="", id="main_input", classes="main-input")

    def on_mount(self) -> None:
        self._apply_accent()

    def set_provider(self, provider: str, token_count: int) -> None:
        self._provider = provider
        self._token_count = token_count
        self._apply_accent()

    def _apply_accent(self) -> None:
        accent = get_accent(self._provider)
        self.styles.border = ("round", accent)

        self.border_title = Text(f" {self._provider} ", style=f"bold {accent}")

        if self._token_count >= 1000:
            tok_str = f"~{self._token_count / 1000:.1f}k tokens"
        else:
            tok_str = f"~{self._token_count} tokens"
        self.border_subtitle = Text(f" {tok_str} ", style=f"dim {PALETTE.text_dim}")

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
        t.append("\u25b6\u25b6 ", style=f"bold {accent}")
        t.append(f"{self._mode} mode", style=f"bold {accent}")
        t.append(" . ", style=f"dim {PALETTE.text_dim}")
        t.append("shift+tab to cycle", style=f"dim {PALETTE.text_dim}")
        return t
