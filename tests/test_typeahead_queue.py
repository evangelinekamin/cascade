"""Type-ahead queue: FIFO, busy on slash lanes, cleared on interrupt."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from textual.message_pump import active_app
from cascade.screens.main import MainScreen


def _screen_with(app):
    screen = MainScreen()
    screen._cmd_handler = MagicMock()
    screen._cmd_handler.is_command.return_value = False
    return screen


def _submit(screen, app, value):
    event = MagicMock()
    event.value = value
    event.text_area = MagicMock()
    token = active_app.set(app)
    try:
        screen.on_chat_text_area_submitted(event)
    finally:
        active_app.reset(token)
    return event


def test_multiple_prompts_queue_fifo_not_overwrite():
    app = MagicMock()
    app.state.is_thinking = True
    screen = _screen_with(app)
    _submit(screen, app, "first")
    _submit(screen, app, "second")
    assert list(screen._queued_prompts) == ["first", "second"]


def test_busy_via_slash_lane_queues_chat_prompt():
    app = MagicMock()
    app.state.is_thinking = False  # no chat turn...
    screen = _screen_with(app)
    screen._active_run = None
    screen._cmd_handler.is_busy.return_value = True  # ...but a /solve lane runs
    _submit(screen, app, "hello")
    assert list(screen._queued_prompts) == ["hello"]


def test_whitespace_only_submit_clears_and_does_nothing():
    app = MagicMock()
    app.state.is_thinking = False
    screen = _screen_with(app)
    screen._active_run = None
    screen._cmd_handler.is_busy.return_value = False
    screen._dispatch_prompt = MagicMock()
    event = _submit(screen, app, "   \n  ")
    event.text_area.load_text.assert_called_once_with("")
    screen._dispatch_prompt.assert_not_called()


def test_interrupt_clears_the_queue():
    from cascade.swarm.lifecycle import RunContext

    app = MagicMock()
    screen = _screen_with(app)
    screen._queued_prompts.append("pending")
    screen._active_run = RunContext(objective="x", workflow="chat")
    screen._reset_cancelled_ui = MagicMock()
    screen._post_system_message = MagicMock()
    screen._interrupt_active_run()
    assert len(screen._queued_prompts) == 0


def test_pull_skips_a_queued_command_leaving_it_for_the_drain():
    # A slash command typed mid-turn must NOT be injected into the model loop as
    # chat text; _pull_queued_prompt returns None and leaves it queued for the
    # completion drain, which dispatches it through the CommandHandler.
    screen = MainScreen()
    screen._cmd_handler = MagicMock()
    screen._cmd_handler.is_command.side_effect = lambda t: t.startswith("/")
    screen._queued_prompts.append("/solve fix the flaky test")

    assert screen._pull_queued_prompt() is None
    assert list(screen._queued_prompts) == ["/solve fix the flaky test"]  # still queued


def test_pull_returns_a_plain_prompt_for_mid_turn_injection():
    screen = MainScreen()
    screen._cmd_handler = MagicMock()
    screen._cmd_handler.is_command.side_effect = lambda t: t.startswith("/")
    screen._reflect_injected_prompt = MagicMock()
    screen._queued_prompts.append("also handle the retry path")

    app = SimpleNamespace(call_from_thread=lambda fn, *a: fn(*a))
    token = active_app.set(app)
    try:
        assert screen._pull_queued_prompt() == "also handle the retry path"
    finally:
        active_app.reset(token)
    assert list(screen._queued_prompts) == []
