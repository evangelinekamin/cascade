"""Cascade wordmark banner.

Renders the "Cascade Logo (Blue)" designed in Claude Design: the ANSI-Shadow
"CASCADE" wordmark with a vertical gradient from light aqua at the top to deep
blue at the bottom, one color per row. The gradient stops live in the theme
palette (``PALETTE.logo_top`` / ``PALETTE.logo_bottom``).
"""

from rich.text import Text

from ..theme import PALETTE

# ANSI-Shadow figlet art for CASCADE. The vertical gradient is applied per row,
# so trailing spaces are significant for alignment -- keep the rows verbatim.
CASCADE_ROWS = [
    " ██████╗ █████╗ ███████╗ ██████╗ █████╗ ██████╗ ███████╗",
    "██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝",
    "██║     ███████║███████╗██║     ███████║██║  ██║█████╗  ",
    "██║     ██╔══██║╚════██║██║     ██╔══██║██║  ██║██╔══╝  ",
    "╚██████╗██║  ██║███████║╚██████╗██║  ██║██████╔╝███████╗",
    " ╚═════╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═════╝ ╚══════╝",
]


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Convert a ``#rrggbb`` string to an ``(r, g, b)`` tuple."""
    h = value.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _vertical_ramp(top: str, bottom: str, n: int) -> list[tuple[int, int, int]]:
    """Interpolate ``n`` RGB stops from ``top`` to ``bottom`` (inclusive)."""
    ar, ag, ab = _hex_to_rgb(top)
    br, bg, bb = _hex_to_rgb(bottom)
    if n <= 1:
        return [(ar, ag, ab)]
    return [
        (
            round(ar + (br - ar) * i / (n - 1)),
            round(ag + (bg - ag) * i / (n - 1)),
            round(ab + (bb - ab) * i / (n - 1)),
        )
        for i in range(n)
    ]


def render_banner(word: str = "CASCADE", indent: int = 1) -> Text:
    """Render the CASCADE wordmark with a top-to-bottom blue gradient.

    The gradient runs from ``PALETTE.logo_top`` (top row) to
    ``PALETTE.logo_bottom`` (bottom row), one color per row. Any word other
    than CASCADE falls back to a plain padded label in the top color.
    """
    rows = CASCADE_ROWS if word.upper() == "CASCADE" else [f"  {word}  "]
    colors = _vertical_ramp(PALETTE.logo_top, PALETTE.logo_bottom, len(rows))

    result = Text()
    for line, (r, g, b) in zip(rows, colors):
        result.append(" " * indent)
        result.append(line, style=f"rgb({r},{g},{b})")
        result.append("\n")
    return result
