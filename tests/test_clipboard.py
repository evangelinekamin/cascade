"""Tests for auto-copy clipboard confirmation formatting."""

from cascade.app import CascadeTUI

_msg = CascadeTUI._copied_message


def test_copied_message_pluralizes_many():
    assert _msg(12) == "Copied 12 characters to clipboard"


def test_copied_message_singular_one():
    assert _msg(1) == "Copied 1 character to clipboard"


def test_copied_message_thousands_separator():
    assert _msg(12345) == "Copied 12,345 characters to clipboard"


# -- StatusBar bottom-right flash (copy confirmation / exit hint) --------------

from cascade.widgets.status_bar import StatusBar


def test_status_bar_flash_message_renders_in_the_corner():
    sb = StatusBar(cwd="~/x", branch="main")
    sb._flash = "Copied 5 characters to clipboard"
    assert "Copied 5 characters to clipboard" in sb.render().plain


def test_status_bar_without_flash_shows_no_flash_text():
    sb = StatusBar(cwd="~/x", branch="main", provider_tokens={})
    assert "Copied" not in sb.render().plain
