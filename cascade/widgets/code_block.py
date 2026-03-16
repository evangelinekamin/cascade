"""Syntax-highlighted code container with box-drawing border.

Language label (accent color) top-left, "copy" top-right.
Code bg #161b22, border #30363d.
"""

from rich.syntax import Syntax
from rich.text import Text
from rich.panel import Panel
from textual.widgets import Static

from ..theme import PALETTE, get_accent


class CodeBlock(Static):
    """Bordered code block with syntax highlighting and line numbers."""

    DEFAULT_CSS = """
    CodeBlock {
        height: auto;
        width: 100%;
        margin: 1 0;
    }
    """

    def __init__(
        self,
        code: str,
        language: str = "text",
        provider: str = "gemini",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._code = code
        self._language = language
        self._provider = provider

    def render(self) -> Panel | Text:
        try:
            accent = get_accent(self._provider)

            syntax = Syntax(
                self._code or "",
                self._language or "text",
                theme="monokai",
                line_numbers=True,
                word_wrap=False,
                background_color=PALETTE.code_bg,
            )

            title = Text()
            title.append(f" {(self._language or 'text').upper()} ", style=f"bold {accent}")

            subtitle = Text(" copy ", style=f"dim {PALETTE.text_dim}")

            return Panel(
                syntax,
                title=title,
                title_align="left",
                subtitle=subtitle,
                subtitle_align="right",
                border_style=PALETTE.border,
                background=PALETTE.code_bg,
                padding=(0, 1),
                expand=True,
            )
        except Exception:
            return Text(self._code or "")
