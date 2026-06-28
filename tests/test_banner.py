"""Tests for the pyfiglet ASCII art banner."""

from rich.text import Text

from cascade.ui.banner import render_banner, _lerp_color, GRADIENT


def test_render_banner_returns_text():
    """render_banner should return a Rich Text object."""
    result = render_banner()
    assert isinstance(result, Text)


def test_render_banner_default_word():
    """Default word is CASCADE."""
    result = render_banner()
    plain = result.plain
    lines = plain.split("\n")
    non_empty = [line for line in lines if line.strip()]
    assert len(non_empty) >= 3, "Banner should have at least 3 non-empty rows"


def test_render_banner_contains_art_chars():
    """Banner should contain figlet art characters (slashes, pipes, underscores)."""
    result = render_banner()
    plain = result.plain
    # small font uses ASCII art chars like /, \, |, _
    art_chars = set("/\\|_")
    has_art = any(ch in art_chars for ch in plain)
    assert has_art, "Banner should use figlet art characters"


def test_render_banner_custom_word():
    """render_banner should accept a custom word."""
    result = render_banner("CASE")
    assert isinstance(result, Text)
    assert len(result.plain) > 0


def test_lerp_color_boundaries():
    """Interpolation at 0 and 1 should return first and last stops."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    assert _lerp_color(colors, 0.0) == (255, 0, 0)
    assert _lerp_color(colors, 1.0) == (0, 0, 255)


def test_lerp_color_midpoint():
    """Interpolation at 0.5 should return the middle stop."""
    colors = [(0, 0, 0), (100, 100, 100), (200, 200, 200)]
    result = _lerp_color(colors, 0.5)
    assert result == (100, 100, 100)


def test_gradient_has_colors():
    """Gradient should have multiple color stops."""
    assert len(GRADIENT) >= 3


def test_banner_fits_80_columns():
    """Banner should fit within 80 columns."""
    result = render_banner()
    for line in result.plain.split("\n"):
        assert len(line) <= 80, f"Line too wide ({len(line)} chars): {line!r}"
