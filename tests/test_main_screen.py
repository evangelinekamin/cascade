"""Tests for MainScreen helper behavior."""

from unittest.mock import patch

from cascade.screens.main import MainScreen, summarize_user_prompt


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
