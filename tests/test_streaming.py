"""Tests for streaming polish -- pause-driven paragraph breaks."""

from cascade.screens.main import MainScreen

_break = MainScreen._pause_paragraph_break


def test_pause_break_inserts_after_a_sentence_ending_pause():
    assert _break("Next thought", True, 1.0, 0.8) == "\n\nNext thought"
    assert _break("And?", True, 1.2, 0.8) == "\n\nAnd?"


def test_pause_break_skips_a_short_pause():
    assert _break("more", True, 0.3, 0.8) == "more"


def test_pause_break_never_breaks_mid_sentence():
    # the previous text did not end a sentence -> no break even on a long pause
    assert _break("word", False, 2.0, 0.8) == "word"


def test_pause_break_does_not_double_a_leading_newline():
    assert _break("\nalready spaced", True, 2.0, 0.8) == "\nalready spaced"
