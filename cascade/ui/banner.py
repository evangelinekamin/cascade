"""ASCII art banner using pyfiglet with smooth gradient coloring.

Uses the 'small' font for a lightweight, refined look with a
left-to-right gradient from cyan -> purple -> pink. Falls back to
a built-in banner if pyfiglet is unavailable.
"""

import logging

from rich.text import Text

try:
    import pyfiglet
    _HAS_PYFIGLET = True
except ImportError:
    _HAS_PYFIGLET = False

_log = logging.getLogger(__name__)

# Gradient stops: cyan -> purple -> pink
GRADIENT = [
    (0, 212, 229),    # #00d4e5  cyan
    (124, 58, 237),   # #7c3aed  purple
    (229, 90, 155),   # #e55a9b  pink
]

# Preferred figlet font -- lightweight and refined
FONT = "small"

# Built-in fallback keeps the UI readable even when pyfiglet is not installed.
_FALLBACK_CASCADE = [
    r"  ___   _   ___  ___   _   ___  ___ ",
    r" / __| /_\ / __|/ __| /_\ |   \| __|",
    r"| (__ / _ \\__ \| (__ / _ \| |) | _| ",
    r" \___/_/ \_\___/ \___/_/ \_\___/|___|",
]


def _lerp_color(
    colors: list[tuple[int, int, int]], fraction: float,
) -> tuple[int, int, int]:
    """Linearly interpolate across multiple RGB stops."""
    if fraction <= 0:
        return colors[0]
    if fraction >= 1:
        return colors[-1]

    seg_count = len(colors) - 1
    scaled = fraction * seg_count
    idx = int(scaled)
    rem = scaled - idx

    c1 = colors[idx]
    c2 = colors[min(idx + 1, seg_count)]

    return (
        int(c1[0] + (c2[0] - c1[0]) * rem),
        int(c1[1] + (c2[1] - c1[1]) * rem),
        int(c1[2] + (c2[2] - c1[2]) * rem),
    )


def _generate_figlet(word: str) -> list[str]:
    """Generate ASCII art lines using pyfiglet, with fallback."""
    if not _HAS_PYFIGLET:
        _log.warning("pyfiglet not installed -- using built-in fallback banner")
        if word.upper() == "CASCADE":
            return _FALLBACK_CASCADE
        return [f"  {word}  "]
    try:
        raw = pyfiglet.figlet_format(word, font=FONT)
        lines = raw.rstrip("\n").split("\n")
        if lines and any(line.strip() for line in lines):
            return lines
        if word.upper() == "CASCADE":
            return _FALLBACK_CASCADE
        return [f"  {word}  "]
    except Exception as exc:
        _log.warning("pyfiglet rendering failed: %s", exc)
        if word.upper() == "CASCADE":
            return _FALLBACK_CASCADE
        return [f"  {word}  "]


def render_banner(word: str = "CASCADE", indent: int = 1) -> Text:
    """Render the word as gradient-colored ASCII art.

    Uses pyfiglet's ansi_shadow font with a left-to-right gradient.
    Each non-space character gets colored based on its horizontal position.
    """
    lines = _generate_figlet(word)
    max_width = max(len(line) for line in lines) if lines else 1

    result = Text()

    for line in lines:
        result.append(" " * indent)
        for i, ch in enumerate(line):
            if ch == " ":
                result.append(" ")
            else:
                frac = i / max(max_width - 1, 1)
                r, g, b = _lerp_color(GRADIENT, frac)
                result.append(ch, style=f"bold rgb({r},{g},{b})")
        result.append("\n")

    return result
