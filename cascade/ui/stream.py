"""Streaming response renderer with code block detection.

Processes character-by-character chunks from provider streams, routing
prose lines through the gutter+markdown renderer and code blocks through
the code_block container.
"""

from enum import Enum, auto

from rich.console import Console

from .code_block import render_code_container
from .gutter import render_gutter_line
from .markdown import render_markdown_line
from .theme import ProviderTheme


class _State(Enum):
    PROSE = auto()
    CODE_BLOCK = auto()


class StreamRenderer:
    """Stateful streaming renderer that handles arbitrary chunk boundaries.

    Usage:
        renderer = StreamRenderer(theme, console)
        for chunk in provider.stream_single(prompt, system):
            renderer.feed(chunk)
        renderer.finish()
    """

    def __init__(self, theme: ProviderTheme, console: Console):
        self._theme = theme
        self._console = console
        self._state = _State.PROSE
        self._line_buf = ""
        self._code_buf = ""
        self._code_lang = ""
        self._first_line = True
        self._fence_count = 0

    @property
    def is_first_line(self) -> bool:
        return self._first_line

    def feed(self, chunk: str) -> None:
        """Feed a chunk of text from the streaming provider."""
        for ch in chunk:
            self._process_char(ch)

    def finish(self) -> None:
        """Flush any remaining buffered content."""
        if self._state == _State.CODE_BLOCK:
            # Unterminated code block -- render what we have
            if self._code_buf:
                render_code_container(
                    self._code_buf.rstrip("\n"),
                    self._code_lang,
                    self._console,
                )
            self._code_buf = ""
            self._state = _State.PROSE

        # Flush remaining prose line
        if self._line_buf.strip():
            self._emit_prose_line(self._line_buf)
            self._line_buf = ""

    def _process_char(self, ch: str) -> None:
        if self._state == _State.PROSE:
            self._line_buf += ch
            if ch == "\n":
                line = self._line_buf.rstrip("\n")
                # Check for code fence opening
                stripped = line.strip()
                if stripped.startswith("```"):
                    self._code_lang = stripped[3:].strip()
                    self._state = _State.CODE_BLOCK
                    self._code_buf = ""
                    self._line_buf = ""
                else:
                    self._emit_prose_line(line)
                    self._line_buf = ""
        else:
            # CODE_BLOCK state
            self._line_buf += ch
            if ch == "\n":
                line = self._line_buf.rstrip("\n")
                if line.strip() == "```":
                    # End of code block
                    render_code_container(
                        self._code_buf.rstrip("\n"),
                        self._code_lang,
                        self._console,
                    )
                    self._code_buf = ""
                    self._code_lang = ""
                    self._state = _State.PROSE
                else:
                    self._code_buf += self._line_buf
                self._line_buf = ""

    def _emit_prose_line(self, line: str) -> None:
        """Render a single prose line through the gutter."""
        styled = render_markdown_line(line)
        guttered = render_gutter_line(
            styled,
            abbreviation=self._theme.abbreviation,
            accent=self._theme.accent,
            first=self._first_line,
        )
        self._console.print(guttered)
        self._first_line = False
