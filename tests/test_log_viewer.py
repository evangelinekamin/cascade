"""Tests for the /solve log viewer screen and command."""

from unittest.mock import MagicMock

from cascade.screens.log_viewer import LogViewerScreen, _colorize
from cascade.commands import CommandHandler, COMMANDS


def test_colorize_diff_and_plain_lines():
    assert _colorize("+added line").style == "green"
    assert _colorize("-removed line").style == "red"
    # +++ / --- are diff headers, not additions/removals
    assert _colorize("+++ b/file.py").style == "bold cyan"
    assert _colorize("--- a/file.py").style == "bold cyan"
    assert _colorize("[editing] deepseek").style == "dim"
    assert _colorize("plain text").style == ""


def test_log_viewer_holds_title_and_lines():
    screen = LogViewerScreen("/solve PASSED", ["one", "+two", "-three"])
    assert screen._title == "/solve PASSED"
    assert screen._lines == ["one", "+two", "-three"]


def test_log_command_registered():
    assert any(c.name == "log" for c in COMMANDS)


def test_cmd_log_with_no_solve_posts_hint():
    handler = CommandHandler(MagicMock())
    posted: list[str] = []
    handler._post_system = lambda text, **kwargs: posted.append(text)

    handler._cmd_log([])

    assert posted and "No /solve log" in posted[0]
