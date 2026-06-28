"""Regression tests for TUI session persistence and lifecycle glue."""

from unittest.mock import MagicMock

import pytest
from textual.message_pump import active_app

from cascade.app import CascadeTUI
from cascade.commands import CommandHandler
from cascade.config import ConfigManager
from cascade.hooks import HookEvent
from cascade.history import BranchingSession, HistoryDB
from cascade.screens.main import MainScreen, WelcomeHeader
from cascade.state import CascadeState, ThinkingChanged
from cascade.widgets.header import ProviderGhostTable
from cascade.widgets.input_frame import InputFrame
from cascade.widgets.message import ChatHistory
from cascade.widgets.status_bar import StatusBar


@pytest.fixture
def tui_app(tmp_path, monkeypatch):
    db_path = tmp_path / "history.db"
    monkeypatch.setattr("cascade.app.HistoryDB", lambda: HistoryDB(db_path=str(db_path)))
    app = CascadeTUI(cli_app=None)
    yield app
    app.db.close()


def test_ensure_session_keeps_state_and_db_ids_in_sync(tui_app):
    session = tui_app.ensure_session()
    assert session["id"] == tui_app.state.session_id


def test_tui_prefers_first_configured_provider_when_default_missing(tmp_path, monkeypatch):
    db_path = tmp_path / "history.db"
    monkeypatch.setattr("cascade.app.HistoryDB", lambda: HistoryDB(db_path=str(db_path)))

    cli_app = MagicMock()
    cli_app.config.get_default_provider.return_value = "gemini"
    cli_app.providers = {
        "claude": MagicMock(),
        "openrouter": MagicMock(),
    }

    app = CascadeTUI(cli_app=cli_app)
    try:
        assert app.state.active_provider == "claude"
        assert app.state.mode == "plan"
    finally:
        app.db.close()


def test_record_message_persists_parent_chain_via_branching(tui_app):
    tui_app.record_message("user", "Hello")
    tui_app.record_message("claude", "Hi there", token_count=12)

    messages = tui_app.db.get_session_messages(tui_app.state.session_id)
    assert [msg["role"] for msg in messages] == ["user", "claude"]
    assert messages[0]["metadata"]["parent_id"] is None
    assert messages[1]["metadata"]["parent_id"] == messages[0]["id"]

    path = tui_app.get_branching_session().get_path_to_leaf()
    assert [msg["id"] for msg in path] == [messages[0]["id"], messages[1]["id"]]


def test_tui_window_title_tracks_busy_activity(tui_app):
    tui_app.console.set_window_title = MagicMock()
    tui_app.state.active_provider = "claude"
    tui_app.state.set_session_id("night-river")

    assert tui_app._formatted_window_title() == "cascade . claude . night-river"

    tui_app.start_title_activity("chat", "claude", "thinking")
    busy_title = tui_app._formatted_window_title()
    assert "cascade . claude . night-river" in busy_title
    assert "thinking" in busy_title

    tui_app.stop_title_activity("chat")
    assert tui_app._formatted_window_title() == "cascade . claude . night-river"


def test_tui_thinking_event_updates_window_title_state(tui_app):
    tui_app.console.set_window_title = MagicMock()
    tui_app.state.active_provider = "gemini"
    tui_app.state.set_session_id("cedar-pulse")

    tui_app.on_thinking_changed(ThinkingChanged("openrouter", True, "routing"))
    title = tui_app._formatted_window_title()
    assert "routing" in title
    assert "openrouter" in title

    tui_app.on_thinking_changed(ThinkingChanged("openrouter", False, ""))
    assert tui_app._formatted_window_title() == "cascade . gemini . cedar-pulse"


class _FakeApp:
    def __init__(self, db, screen, cli_app):
        self.db = db
        self.screen = screen
        self.cli_app = cli_app
        self.state = CascadeState()
        self._db_session = None

    def adopt_session(self, session: dict) -> dict:
        self._db_session = session
        self.state.set_session_id(session["id"])
        return session

    def get_branching_session(self) -> BranchingSession:
        return BranchingSession(self.db, self._db_session["id"])

    def notify(self, _text: str) -> None:
        return None


class _FakeCommandApp:
    def __init__(self) -> None:
        self.state = CascadeState()
        self.screen = MagicMock()
        self.screen.query_one.return_value = MagicMock()
        self._title_events: list[tuple[str, str, str]] = []

    def start_title_activity(self, source: str, provider: str, label: str) -> None:
        self._title_events.append(("start", provider, label))

    def update_title_activity(self, source: str, label: str, provider: str | None = None) -> None:
        self._title_events.append(("update", provider or "", label))

    def stop_title_activity(self, source: str) -> None:
        self._title_events.append(("stop", "", source))

    def notify(self, _text: str) -> None:
        return None


def test_resume_resets_state_preserves_roles_and_emits_hook(tmp_path):
    db = HistoryDB(db_path=str(tmp_path / "resume.db"))
    session = db.create_session(provider="claude", title="Saved Session")
    db.add_message(session["id"], role="user", content="Hello")
    db.add_message(
        session["id"],
        role="assistant",
        content="Hi from Claude",
        token_count=12,
        metadata={"provider": "claude"},
    )
    db.add_message(session["id"], role="gemini", content="Cross-model answer", token_count=20)

    chat = MagicMock()
    status = MagicMock()
    input_frame = MagicMock()
    header = MagicMock()
    screen = MagicMock()
    screen._header_visible = True
    screen._cross_model_summary = "stale summary"
    screen._turns_since_summary = 9
    screen._summary_compaction_running = True
    screen.query_one.side_effect = lambda selector: {
        ChatHistory: chat,
        StatusBar: status,
        InputFrame: input_frame,
        WelcomeHeader: header,
    }[selector]

    cli_app = MagicMock()
    cli_app.providers = {"claude": MagicMock()}
    cli_app.hook_runner.emit = MagicMock()
    app = _FakeApp(db, screen, cli_app)
    app.state.add_message("you", "stale message")
    app.state.provider_tokens["claude"] = 999
    app.state.total_tokens = 999

    handler = CommandHandler(app)
    posted = []
    handler._post_system = lambda text: posted.append(text)

    handler._cmd_resume([session["id"]])

    assert app.state.session_id == session["id"]
    assert app.state.active_provider == "claude"
    assert app.state.mode == "plan"
    assert [msg.role for msg in app.state.messages] == ["you", "claude", "gemini"]
    assert [msg.content for msg in app.state.messages] == [
        "Hello",
        "Hi from Claude",
        "Cross-model answer",
    ]
    assert app.state.total_tokens == 32
    assert app.state.provider_tokens["claude"] == 12
    assert app.state.provider_tokens["gemini"] == 20
    assert screen._cross_model_summary == ""
    assert screen._turns_since_summary == 0
    assert screen._summary_compaction_running is False
    chat.remove_children.assert_called_once()
    status.update_tokens.assert_called_once_with(app.state.provider_tokens)
    assert input_frame.active_provider == "claude"
    assert input_frame.mode == "plan"
    assert input_frame.token_count == 32
    assert header.display is False
    assert posted[-1] == "Resumed session: Saved Session (3 messages)"
    assert cli_app.hook_runner.emit.call_args.args[0] == HookEvent.SESSION_RESUME
    assert cli_app.hook_runner.emit.call_args.args[1].session_id == session["id"]

    db.close()


def test_main_screen_exit_emits_on_exit_once():
    hook_runner = MagicMock()
    cli_app = MagicMock()
    cli_app.hook_runner = hook_runner

    app = MagicMock()
    app.cli_app = cli_app
    app.state.session_id = "state-session"
    app.state.elapsed = 30.0
    app.state.message_count = 3
    app.state.response_count = 2
    app.state.provider_tokens = {"claude": 12}
    app._db_session = {"id": "db-session"}
    app.push_screen = MagicMock()

    screen = MainScreen()
    screen._active_provider = "claude"
    screen._mode = "design"

    token = active_app.set(app)
    try:
        screen.action_exit_app()
        screen.action_exit_app()
    finally:
        active_app.reset(token)

    assert hook_runner.emit.call_count == 1
    assert hook_runner.emit.call_args.args[0] == HookEvent.ON_EXIT
    assert hook_runner.emit.call_args.args[1].session_id == "db-session"
    app.push_screen.assert_called()


def test_main_screen_rejects_new_prompt_while_thinking():
    app = MagicMock()
    app.state.is_thinking = True
    app.notify = MagicMock()

    screen = MainScreen()
    screen._cmd_handler = MagicMock()

    event = MagicMock()
    event.value = "hello"
    event.input = MagicMock()
    event.input._pending_paste = None

    token = active_app.set(app)
    try:
        screen.on_input_submitted(event)
    finally:
        active_app.reset(token)

    app.notify.assert_called_once_with("Wait for the current response to finish.")
    screen._cmd_handler.handle.assert_not_called()


def test_main_screen_mouse_scroll_routes_to_chat():
    screen = MainScreen()
    chat = MagicMock()
    screen.query_one = MagicMock(return_value=chat)

    up_event = MagicMock()
    down_event = MagicMock()

    screen.on_mouse_scroll_up(up_event)
    screen.on_mouse_scroll_down(down_event)

    chat.scroll_relative.assert_any_call(y=-6, animate=False, force=True)
    chat.scroll_relative.assert_any_call(y=6, animate=False, force=True)
    up_event.stop.assert_called_once()
    up_event.prevent_default.assert_called_once()
    down_event.stop.assert_called_once()
    down_event.prevent_default.assert_called_once()


def test_chat_history_mouse_scroll_stops_event():
    chat = ChatHistory()
    chat.scroll_relative = MagicMock()

    up_event = MagicMock()
    down_event = MagicMock()

    chat.on_mouse_scroll_up(up_event)
    chat.on_mouse_scroll_down(down_event)

    chat.scroll_relative.assert_any_call(y=-6, animate=False, force=True)
    chat.scroll_relative.assert_any_call(y=6, animate=False, force=True)
    up_event.stop.assert_called_once()
    up_event.prevent_default.assert_called_once()
    down_event.stop.assert_called_once()
    down_event.prevent_default.assert_called_once()


def test_command_progress_handle_updates_title_activity():
    app = _FakeCommandApp()
    handler = CommandHandler(app)

    handle = handler._mount_progress_indicator("claude queued | openai queued")
    handler._set_progress_indicator_label(handle, "claude done | openai running")
    handler._clear_progress_indicator(handle)

    assert app._title_events[0] == ("start", app.state.active_provider, "claude queued | openai queued")
    assert app._title_events[1] == (
        "update",
        app.state.active_provider,
        "claude done | openai running",
    )
    assert app._title_events[2][0] == "stop"


def test_export_preserves_cross_model_provider_labels(tmp_path, monkeypatch):
    db = HistoryDB(db_path=str(tmp_path / "export.db"))
    session = db.create_session(
        provider="gemini",
        model="gemini-model",
        title="Export Session",
    )
    db.add_message(session["id"], role="user", content="Hello")
    db.add_message(
        session["id"],
        role="gemini",
        content="Hi from Gemini",
        token_count=12,
        metadata={"provider": "gemini"},
    )
    db.add_message(
        session["id"],
        role="claude",
        content="Cross-model answer",
        token_count=20,
        metadata={"provider": "claude"},
    )

    app = _FakeApp(db, MagicMock(), cli_app=None)
    app._db_session = session
    app.state.set_session_id(session["id"])

    handler = CommandHandler(app)
    posted = []
    handler._post_system = lambda text: posted.append(text)

    monkeypatch.chdir(tmp_path)
    handler._cmd_export([])

    export_path = tmp_path / f"cascade-session-{session['id']}.md"
    content = export_path.read_text(encoding="utf-8")

    assert "## USER (" in content
    assert "## gemini-model (" in content
    assert "## claude (" in content
    assert "Cross-model answer" in content
    assert posted[-1] == f"Exported 3 messages to {export_path}"

    db.close()


def test_model_command_rejects_unconfigured_provider():
    app = MagicMock()
    app.cli_app = MagicMock()
    app.cli_app.providers = {"claude": MagicMock()}
    app.state = CascadeState()
    app.screen = MagicMock()
    app.notify = MagicMock()

    handler = CommandHandler(app)
    handler._cmd_model(["openai"])

    app.notify.assert_called_once_with(
        "Provider 'openai' is not configured. Available: claude"
    )
    assert app.state.active_provider == "gemini"


def test_mode_command_rejects_mode_for_missing_provider():
    app = MagicMock()
    app.cli_app = MagicMock()
    app.cli_app.providers = {"claude": MagicMock()}
    app.state = CascadeState()
    app.screen = MagicMock()
    app.notify = MagicMock()

    handler = CommandHandler(app)
    handler._cmd_mode(["build"])

    app.notify.assert_called_once_with(
        "Mode 'build' is unavailable because provider 'openai' is not configured. "
        "Available modes: plan"
    )
    assert app.state.mode == "design"


def test_mode_command_uses_configured_mode_provider_and_model(tmp_path):
    config = ConfigManager(str(tmp_path / "config.yaml"))
    config.data["modes"]["design"]["provider"] = "openrouter"
    config.data["modes"]["design"]["model"] = "kwaipilot/kat-coder-pro-v2"

    openrouter = MagicMock()
    openrouter.config.model = "qwen/qwen3.5-9b"

    app = MagicMock()
    app.cli_app = MagicMock()
    app.cli_app.providers = {"openrouter": openrouter}
    app.cli_app.config = config
    app.state = CascadeState()
    app.screen = MagicMock()
    app.notify = MagicMock()

    handler = CommandHandler(app)
    handler._cmd_mode(["design"])

    assert app.state.active_provider == "openrouter"
    assert app.state.mode == "design"
    assert openrouter.config.model == "kwaipilot/kat-coder-pro-v2"
    app.notify.assert_called_once_with("Switched to design mode")


def test_mode_provider_command_applies_immediately_for_active_mode(tmp_path):
    config = ConfigManager(str(tmp_path / "config.yaml"))
    claude = MagicMock()
    claude.config.model = "claude-sonnet-4-6"
    openrouter = MagicMock()
    openrouter.config.model = "qwen/qwen3.5-9b"

    app = MagicMock()
    app.cli_app = MagicMock()
    app.cli_app.providers = {"claude": claude, "openrouter": openrouter}
    app.cli_app.config = config
    app.state = CascadeState()
    app.state.set_provider("claude", "plan")
    app.screen = MagicMock()
    app.notify = MagicMock()

    handler = CommandHandler(app)
    handler._cmd_mode_provider(["plan", "openrouter"])

    assert config.data["modes"]["plan"]["provider"] == "openrouter"
    assert app.state.active_provider == "openrouter"
    assert app.state.mode == "plan"
    app.notify.assert_called_once_with("plan mode now uses openrouter")


def test_mode_model_command_applies_immediately_for_active_mode(tmp_path):
    config = ConfigManager(str(tmp_path / "config.yaml"))
    config.data["modes"]["test"]["provider"] = "openrouter"
    openrouter = MagicMock()
    openrouter.config.model = "qwen/qwen3.5-9b"

    app = MagicMock()
    app.cli_app = MagicMock()
    app.cli_app.providers = {"openrouter": openrouter}
    app.cli_app.config = config
    app.state = CascadeState()
    app.state.set_provider("openrouter", "test")
    app.screen = MagicMock()
    app.notify = MagicMock()

    handler = CommandHandler(app)
    handler._cmd_mode_model(["test", "kwaipilot/kat-coder-pro-v2"])

    assert config.data["modes"]["test"]["model"] == "kwaipilot/kat-coder-pro-v2"
    assert openrouter.config.model == "kwaipilot/kat-coder-pro-v2"
    app.notify.assert_called_once_with("test mode now uses kwaipilot/kat-coder-pro-v2")


def test_cycle_mode_skips_unconfigured_providers():
    screen = MainScreen(
        active_provider="claude",
        mode="plan",
        providers={"claude": MagicMock(), "openrouter": MagicMock()},
    )
    app = MagicMock()
    app.state = CascadeState()
    app.state.set_provider("claude", "plan")
    app.get_branching_session.return_value = MagicMock()

    input_frame = MagicMock()
    ghost = MagicMock()
    chat = MagicMock()
    screen.query_one = MagicMock(side_effect=lambda selector: {
        InputFrame: input_frame,
        ProviderGhostTable: ghost,
        ChatHistory: chat,
    }[selector])

    token = active_app.set(app)
    try:
        screen.action_cycle_mode()
    finally:
        active_app.reset(token)

    assert screen._active_provider == "openrouter"
    assert screen._mode == "test"
    assert app.state.active_provider == "openrouter"
    assert app.state.mode == "test"
    app.get_branching_session.return_value.create_branch.assert_called_once_with(
        label="claude->openrouter",
        provider="openrouter",
    )


def test_provider_ghost_table_stringifies_non_string_models():
    provider = MagicMock()
    provider.config.model = 123
    table = ProviderGhostTable(
        providers={"claude": provider},
        active_provider="claude",
    )

    rendered = table.render()

    assert "123" in rendered.plain
