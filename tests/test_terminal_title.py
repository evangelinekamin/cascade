"""The terminal tab title: what cascade writes to the emulator, and when.

Textual's ``App.console`` writes into a null file, so Rich's ``set_window_title``
never reached the emulator -- the tab kept whatever the shell had left there
("Debian"). These lock the escape onto the stream the driver owns, the workspace
context that tells two cascade tabs apart, and the clear on exit.
"""

import pytest

import cascade.app as app_mod
from cascade.app import CascadeTUI, _emit_terminal_title
from cascade.history import HistoryDB
from cascade.screens.main import MainScreen
from cascade.widgets.status_bar import StatusBar


class _FakeStream:
    """Minimal stdio stand-in that records what was written to it."""

    def __init__(self, tty: bool = True) -> None:
        self.written: list[str] = []
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty

    def write(self, data: str) -> int:
        self.written.append(data)
        return len(data)

    def flush(self) -> None:
        return None


@pytest.fixture
def stream(monkeypatch):
    fake = _FakeStream()
    monkeypatch.setattr(app_mod.sys, "__stderr__", fake)
    monkeypatch.setenv("TERM", "xterm-256color")
    return fake


@pytest.fixture
def tui_app(tmp_path, monkeypatch):
    db_path = tmp_path / "history.db"
    monkeypatch.setattr("cascade.app.HistoryDB", lambda: HistoryDB(db_path=str(db_path)))
    app = CascadeTUI(cli_app=None)
    yield app
    app.db.close()


def test_emit_writes_an_osc_title_to_the_terminal_stream(stream):
    assert _emit_terminal_title("cascade . build . aqua") is True
    assert stream.written == ["\x1b]0;cascade . build . aqua\x07"]


def test_emit_is_a_no_op_off_a_terminal(monkeypatch):
    fake = _FakeStream(tty=False)
    monkeypatch.setattr(app_mod.sys, "__stderr__", fake)
    monkeypatch.setenv("TERM", "xterm-256color")

    assert _emit_terminal_title("cascade") is False
    assert fake.written == []


def test_emit_is_a_no_op_on_a_dumb_terminal(stream, monkeypatch):
    monkeypatch.setenv("TERM", "dumb")

    assert _emit_terminal_title("cascade") is False
    assert stream.written == []


def test_emit_strips_control_characters_from_the_title(stream):
    # Activity labels carry model-authored text; an embedded BEL would close the
    # OSC string early and let the rest run as terminal commands.
    _emit_terminal_title("cascade\x07\x1b]0;pwned\x07 . build")

    assert stream.written == ["\x1b]0;cascade]0;pwned . build\x07"]


def test_terminal_title_leads_with_the_project(tui_app, stream):
    tui_app.state.cwd = "/home/eve/Projects/aqua"
    tui_app.state.mode = "build"
    tui_app.state.active_provider = "claude"
    tui_app.state.set_session_id("night-river")

    # Project first: tabs truncate on the right, so two cascade tabs in
    # different repos must stay distinguishable in ~20 visible characters.
    assert tui_app._terminal_window_title() == (
        "aqua . build . cascade . claude . night-river"
    )
    assert tui_app._terminal_window_title().startswith("aqua")


def test_terminal_title_appends_activity_without_a_spinner(tui_app, stream):
    tui_app.state.cwd = "/home/eve/Projects/aqua"
    tui_app.state.mode = "build"
    tui_app.state.active_provider = "claude"
    tui_app.state.set_session_id("night-river")

    tui_app.start_title_activity("chat", "claude", "thinking")
    title = tui_app._terminal_window_title()

    # No animated frame: the title re-syncs on every 10Hz tick, and an
    # animated tab relayouts the strip constantly (flicker while scanning).
    assert title[0] not in app_mod._TITLE_SPINNER_FRAMES
    assert title == "aqua . build . cascade . claude . night-river . thinking"


def test_repeat_syncs_do_not_rewrite_the_tab(tui_app, stream):
    tui_app.state.cwd = "/home/eve/Projects/aqua"
    tui_app.state.mode = "build"
    tui_app.state.active_provider = "claude"
    tui_app.state.set_session_id("night-river")

    tui_app._sync_window_title()
    tui_app._sync_window_title()
    tui_app._sync_window_title()
    assert len(stream.written) == 1  # memoized: only real changes reach the tab

    tui_app.state.mode = "test"
    tui_app._sync_window_title()
    assert len(stream.written) == 2


def test_sync_window_title_reaches_the_terminal(tui_app, stream):
    tui_app.state.cwd = "/home/eve/Projects/aqua"
    tui_app.state.mode = "design"
    tui_app.state.active_provider = "gemini"
    tui_app.state.set_session_id("cedar-pulse")

    tui_app._sync_window_title()

    assert stream.written == [
        "\x1b]0;aqua . design . cascade . gemini . cedar-pulse\x07"
    ]


def test_provider_and_mode_change_retitles_the_tab(tui_app, stream):
    tui_app.state.cwd = "/home/eve/Projects/aqua"
    tui_app.state.set_session_id("cedar-pulse")
    tui_app.state.active_provider = "openai"
    tui_app.state.mode = "build"

    tui_app.on_provider_changed(app_mod.ProviderChanged("openai", "build"))

    assert stream.written[-1] == (
        "\x1b]0;aqua . build . cascade . openai . cedar-pulse\x07"
    )


@pytest.mark.asyncio
async def test_mounted_app_titles_the_tab_and_clears_it_on_exit(tmp_path, monkeypatch):
    """Render the real app: mount must title the tab, unmount must hand it back."""
    db_path = tmp_path / "history.db"
    monkeypatch.setattr("cascade.app.HistoryDB", lambda: HistoryDB(db_path=str(db_path)))
    emitted: list[str] = []
    monkeypatch.setattr(app_mod, "_emit_terminal_title", lambda title: emitted.append(title))

    app = CascadeTUI(cli_app=None)
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            # The real screen is mounted and painted, not just constructed.
            assert isinstance(app.screen, MainScreen)
            assert app.screen.query_one(StatusBar).render() is not None
            assert emitted, "mounting the app did not title the terminal"
            assert "cascade" in emitted[-1]
            assert app.state.session_id in emitted[-1]
    finally:
        app.db.close()

    # Nothing of the session may outlive it in the tab strip.
    assert emitted[-1] == ""
