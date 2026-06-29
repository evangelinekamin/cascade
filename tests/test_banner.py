"""Tests for the CASCADE wordmark banner (ANSI-Shadow, vertical blue gradient)."""

from rich.text import Text

from cascade.ui.banner import (
    render_banner,
    _vertical_ramp,
    _hex_to_rgb,
    CASCADE_ROWS,
)
from cascade.theme import PALETTE


def test_render_banner_returns_text():
    assert isinstance(render_banner(), Text)


def test_render_banner_default_word_has_six_rows():
    plain = render_banner().plain
    non_empty = [line for line in plain.split("\n") if line.strip()]
    assert len(non_empty) == len(CASCADE_ROWS) == 6


def test_render_banner_contains_block_art_chars():
    """Banner uses ANSI-Shadow block and box-drawing characters."""
    plain = render_banner().plain
    assert "█" in plain
    assert any(ch in plain for ch in "╗╝║═╔╚")


def test_render_banner_custom_word():
    result = render_banner("CASE")
    assert isinstance(result, Text)
    assert "CASE" in result.plain


def test_vertical_ramp_boundaries():
    """First and last stops equal the top and bottom colors."""
    ramp = _vertical_ramp("#a5e8f0", "#1e4fa8", 6)
    assert ramp[0] == _hex_to_rgb("#a5e8f0")
    assert ramp[-1] == _hex_to_rgb("#1e4fa8")
    assert len(ramp) == 6


def test_vertical_ramp_single_stop():
    assert _vertical_ramp("#a5e8f0", "#1e4fa8", 1) == [_hex_to_rgb("#a5e8f0")]


def test_banner_uses_theme_logo_colors():
    """The default banner ramps between the palette's logo colors."""
    colors = _vertical_ramp(PALETTE.logo_top, PALETTE.logo_bottom, 6)
    assert colors[0] == _hex_to_rgb(PALETTE.logo_top)
    assert colors[-1] == _hex_to_rgb(PALETTE.logo_bottom)


def test_banner_fits_80_columns():
    for line in render_banner().plain.split("\n"):
        assert len(line) <= 80, f"Line too wide ({len(line)}): {line!r}"
