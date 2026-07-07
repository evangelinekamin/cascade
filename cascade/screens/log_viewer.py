"""Full-screen scrollable viewer for the last /solve run (activity + verified diff).

Mouse is disabled app-wide, so this is opened by the ``/log`` command and
dismissed with escape. A ``RichLog`` pins the newest output to the bottom and
scrolls, giving the full activity feed and the verified diff a readable home
instead of flooding the chat with a wall of text.
"""

from rich.text import Text
from textual.screen import Screen
from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import RichLog, Static


def _colorize(line: str) -> Text:
    """Light colouring so the verified diff and activity read at a glance."""
    if line.startswith("+") and not line.startswith("+++"):
        return Text(line, style="green")
    if line.startswith("-") and not line.startswith("---"):
        return Text(line, style="red")
    if line.startswith(("@@", "diff --git", "---", "+++", "== ")):
        return Text(line, style="bold cyan")
    if line.startswith("["):  # progress lines: [editing] ..., [verifying] ...
        return Text(line, style="dim")
    return Text(line)


class LogViewerScreen(Screen):
    """Scrollable log of the most recent /solve run."""

    DEFAULT_CSS = """
    LogViewerScreen #log-title { padding: 1 2 0 2; text-style: bold; }
    LogViewerScreen #log-hint  { padding: 0 2 1 2; text-style: dim; }
    LogViewerScreen #log-body  { height: 1fr; padding: 0 2; }
    """

    BINDINGS = [
        Binding("escape", "dismiss_log", "Close", show=True),
        Binding("q", "dismiss_log", "Close", show=False),
    ]

    def __init__(self, title: str, lines: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._lines = lines

    def compose(self) -> ComposeResult:
        yield Static(self._title, id="log-title")
        yield Static("scroll: PgUp / PgDn / Home / End    esc to close", id="log-hint")
        yield RichLog(
            highlight=False, markup=False, wrap=True, auto_scroll=True, id="log-body"
        )

    def on_mount(self) -> None:
        body = self.query_one("#log-body", RichLog)
        for line in self._lines:
            body.write(_colorize(line))
        body.scroll_end(animate=False)

    def action_dismiss_log(self) -> None:
        self.app.pop_screen()
