"""Session picker modal: pick a recent chat in this directory to resume.

/resume with no id opens this instead of printing a usage line -- the word-pair
ids are not something anyone memorizes. Sessions are filtered to the current
working directory (plus unknown-dir sessions, which stay reachable everywhere).
Dismisses with the chosen session id, or None on cancel.
"""

from datetime import datetime, timezone

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from ..theme import PALETTE, get_accent


def _relative_time(iso: str, now: datetime) -> str:
    """Compact 'time ago' for a session's ISO-8601 updated_at timestamp."""
    if not iso:
        return "unknown"
    try:
        ts = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return "unknown"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    secs = max(int((now - ts).total_seconds()), 0)
    if secs < 60:
        return "just now"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 7:
        return f"{days}d ago"
    weeks = days // 7
    if weeks < 5:
        return f"{weeks}w ago"
    return ts.date().isoformat()


def _format_row(session: dict, now: datetime) -> Text:
    """Two-line option label: title, then time / message count / model."""
    title = (session.get("title") or "").strip() or "(untitled)"
    if len(title) > 56:
        title = title[:55] + "…"
    count = int(session.get("message_count") or 0)
    provider = str(session.get("provider") or "")
    label = str(session.get("model") or "") or provider
    accent = get_accent(provider) if provider else PALETTE.text_dim

    text = Text()
    text.append(title, style=f"bold {PALETTE.text_bright}")
    text.append("\n")
    text.append(_relative_time(session.get("updated_at", ""), now), style=PALETTE.text_dim)
    text.append("  ·  ", style=PALETTE.text_muted)
    text.append(f"{count} msg{'' if count == 1 else 's'}", style=PALETTE.text_dim)
    if label:
        text.append("  ·  ", style=PALETTE.text_muted)
        text.append(label, style=accent)
    return text


class _SessionList(OptionList):
    """OptionList with vim j/k alongside the built-in arrow navigation."""

    BINDINGS = [
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]


class SessionPickerScreen(ModalScreen[str | None]):
    """Modal listing recent sessions; dismisses the chosen id or None."""

    DEFAULT_CSS = """
    SessionPickerScreen {
        align: center middle;
    }
    SessionPickerScreen > Vertical {
        width: 80;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        border: solid #00d4e5;
        background: #0d1117;
    }
    SessionPickerScreen .picker-header {
        padding: 0 0 1 0;
    }
    SessionPickerScreen .picker-hint {
        padding: 1 0 0 0;
    }
    SessionPickerScreen OptionList {
        height: auto;
        max-height: 20;
        background: #0d1117;
        border: none;
        padding: 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, sessions: list[dict], *, now: datetime | None = None) -> None:
        super().__init__()
        self._sessions = list(sessions)
        self._now = now or datetime.now(timezone.utc)

    def compose(self) -> ComposeResult:
        header = Text()
        header.append("resume ", style=f"bold {PALETTE.cyan}")
        header.append("· recent chats in this directory", style=PALETTE.text_dim)

        options = [
            Option(_format_row(s, self._now), id=s["id"]) for s in self._sessions
        ]
        hint = Text(
            "↑/↓ or j/k · enter resume · esc cancel",
            style=f"dim {PALETTE.text_dim}",
        )

        with Vertical():
            yield Static(header, classes="picker-header")
            yield _SessionList(*options, id="session-picker-list")
            yield Static(hint, classes="picker-hint")

    def on_mount(self) -> None:
        # Focus the list so arrows / j / k / enter drive it immediately.
        self.query_one(_SessionList).focus()

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)
