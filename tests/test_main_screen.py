"""Tests for MainScreen helper behavior."""

from unittest.mock import MagicMock, patch

from cascade.hooks import HookResult
from cascade.screens.main import MainScreen, summarize_user_prompt
from cascade.swarm.lifecycle import RunContext


def test_summarize_user_prompt_single_line_unchanged():
    text = "hello world"
    assert summarize_user_prompt(text) == text


def test_summarize_user_prompt_multiline_collapses():
    text = "line1\nline2\nline3\nline4"
    assert summarize_user_prompt(text) == "[pasted content 1 + 3 lines]"


def test_stream_chunk_coalescing_keeps_fast_first_token_and_batches_burst():
    with patch(
        "cascade.screens.main.time.monotonic",
        side_effect=[0.00, 0.01, 0.02, 0.04],
    ):
        chunks = list(MainScreen._coalesce_stream_chunks(iter(["a", "b", "c", "d"])))

    assert chunks == ["a", "bcd"]


def test_stream_chunk_coalescing_flushes_buffer_on_size_cap():
    large_b = "b" * 800
    large_c = "c" * 400

    with patch(
        "cascade.screens.main.time.monotonic",
        side_effect=[0.00, 0.01, 0.011],
    ):
        chunks = list(
            MainScreen._coalesce_stream_chunks(iter(["a", large_b, large_c]))
        )

    assert chunks == ["a", large_b + large_c]


def test_run_dispatch_rejects_callbacks_from_cancelled_or_superseded_workers():
    screen = MainScreen()
    first = RunContext(objective="first")
    second = RunContext(objective="second")
    seen = []

    screen._active_run = first
    screen._dispatch_for_run(first.id, False, seen.append, ("accepted",))
    screen._active_run = second
    screen._dispatch_for_run(first.id, False, seen.append, ("stale",))
    second.cancel()
    screen._dispatch_for_run(second.id, False, seen.append, ("cancelled",))

    assert seen == ["accepted"]


def test_terminal_run_dispatch_releases_only_the_matching_active_run():
    screen = MainScreen()
    run = RunContext(objective="finish")
    seen = []
    screen._active_run = run

    screen._dispatch_for_run(run.id, True, seen.append, ("done",))

    assert seen == ["done"]
    assert screen._active_run is None


def test_input_hook_completion_dispatches_transformed_prompt():
    screen = MainScreen()
    screen._input_hook_pending = True
    screen._dispatch_ready_prompt = MagicMock()

    screen._finish_input_hook(
        "original",
        HookResult(transformed_value="rewritten"),
        "",
    )

    assert screen._input_hook_pending is False
    screen._dispatch_ready_prompt.assert_called_once_with("rewritten")


def test_input_hook_completion_releases_queue_after_block():
    screen = MainScreen()
    screen._input_hook_pending = True
    screen._post_system_message = MagicMock()
    screen._set_input_locked = MagicMock()
    screen._dispatch_ready_prompt = MagicMock()

    screen._finish_input_hook(
        "original",
        HookResult(block=True, reason="policy"),
        "",
    )

    screen._dispatch_ready_prompt.assert_not_called()
    screen._set_input_locked.assert_called_once_with(False)
    assert "policy" in screen._post_system_message.call_args.args[0]


# -- Ctrl+C interrupt / clear-input / confirm-exit state machine --------------

def _exit_test_screen():
    """A MainScreen with the DOM/timer touchpoints stubbed for Ctrl+C tests."""
    from types import SimpleNamespace

    screen = MainScreen()
    screen._interrupt_active_run = lambda: False
    flashes: list = []
    exited: list = []
    screen.flash_status = lambda msg, *a: flashes.append(msg)
    screen.set_timer = lambda *a, **k: SimpleNamespace(stop=lambda: None)
    screen._perform_exit = lambda: exited.append(True)
    return screen, flashes, exited


def test_ctrl_c_clears_filled_input_before_arming_exit():
    from types import SimpleNamespace

    screen, flashes, exited = _exit_test_screen()
    cleared = {"done": False}
    inp = SimpleNamespace(text="draft text", load_text=lambda v: cleared.update(done=(v == "")))
    screen._input_widget = lambda: inp

    screen.action_exit_app()

    assert cleared["done"] is True
    assert screen._exit_armed is False
    assert exited == []
    assert flashes == []


def test_ctrl_c_twice_on_empty_input_arms_then_exits():
    from types import SimpleNamespace

    screen, flashes, exited = _exit_test_screen()
    screen._input_widget = lambda: SimpleNamespace(text="", load_text=lambda v: None)

    screen.action_exit_app()
    assert screen._exit_armed is True
    assert exited == []
    assert flashes and "exit" in flashes[-1]

    screen.action_exit_app()
    assert exited == [True]
    assert screen._exit_armed is False


def test_ctrl_c_interrupts_active_run_without_arming_exit():
    from types import SimpleNamespace

    screen, _flashes, exited = _exit_test_screen()
    screen._interrupt_active_run = lambda: True
    screen._input_widget = lambda: SimpleNamespace(text="", load_text=lambda v: None)

    screen.action_exit_app()

    assert screen._exit_armed is False
    assert exited == []
