"""Console rendering kept for the one-shot `cascade-cli` and shared banner/theme.

The Rich REPL that once lived here is gone; the live UI is the Textual
TUI in cascade/widgets/ + cascade/screens/.
"""

from .theme import THEME, render_header, render_footer
from .output import render_response, render_comparison, render_thinking, render_error
from .banner import render_banner

__all__ = [
    "THEME",
    "render_header",
    "render_footer",
    "render_response",
    "render_comparison",
    "render_thinking",
    "render_error",
    "render_banner",
]
