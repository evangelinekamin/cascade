"""Tests for MainScreen helper behavior."""

from unittest.mock import patch

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
