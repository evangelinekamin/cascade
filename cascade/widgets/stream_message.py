"""Streaming message widget -- accumulates chunks and renders progressively.

Ports the state machine from cascade/ui/stream.py into Textual widgets.
Detects ```fences to switch between prose and code block rendering.
"""

from enum import Enum, auto

from rich.text import Text
from textual.containers import Vertical
from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Static

from .message import render_content, GutterLabel, GutterSeparator
from .code_block import CodeBlock


class _StreamState(Enum):
    PROSE = auto()
    CODE_BLOCK = auto()


class StreamMessage(Widget):
    """A live-updating message that receives streaming chunks.

    Usage:
        msg = StreamMessage(provider)
        parent.mount(msg)
        for chunk in stream:
            msg.feed(chunk)
        msg.finish()
    """

    DEFAULT_CSS = """
    StreamMessage {
        height: auto;
        width: 100%;
        padding: 0 0 1 0;
        layout: horizontal;
    }

    .stream-body {
        width: 1fr;
        height: auto;
        layout: vertical;
        padding-left: 1;
    }
    """

    def __init__(self, provider: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._provider = provider
        self._state = _StreamState.PROSE
        self._line_buf = ""
        self._code_buf = ""
        self._code_lang = ""
        self._prose_lines: list[str] = []
        self._prose_widget: _ProseBody | None = None
        self._body_column: Vertical | None = None

    def compose(self) -> ComposeResult:
        yield GutterLabel(self._provider)
        yield GutterSeparator()
        with Vertical(classes="stream-body"):
            self._prose_widget = _ProseBody("")
            yield self._prose_widget

    def on_mount(self) -> None:
        try:
            self._body_column = self.query_one(".stream-body", Vertical)
        except Exception:
            self._body_column = None

    def feed(self, chunk: str) -> None:
        """Feed a streaming chunk. Handles arbitrary chunk boundaries."""
        for ch in chunk:
            self._process_char(ch)
        if self._state == _StreamState.PROSE and self._line_buf:
            self._refresh_prose(include_partial=True)

    def finish(self) -> None:
        """Flush any remaining buffered content."""
        if self._state == _StreamState.CODE_BLOCK:
            if self._code_buf:
                self._emit_code_block(self._code_buf.rstrip("\n"), self._code_lang)
            self._code_buf = ""
            self._state = _StreamState.PROSE

        if self._line_buf:
            self._prose_lines.append(self._line_buf)
            self._line_buf = ""
            self._refresh_prose()
        else:
            self._refresh_layout()

    def _process_char(self, ch: str) -> None:
        if self._state == _StreamState.PROSE:
            self._line_buf += ch
            if ch == "\n":
                line = self._line_buf.rstrip("\n")
                stripped = line.strip()
                if stripped.startswith("```"):
                    # Opening fence -- switch to code block mode
                    self._code_lang = stripped[3:].strip()
                    self._state = _StreamState.CODE_BLOCK
                    self._code_buf = ""
                else:
                    self._prose_lines.append(line)
                    self._refresh_prose()
                self._line_buf = ""
        else:
            # CODE_BLOCK state
            self._line_buf += ch
            if ch == "\n":
                line = self._line_buf.rstrip("\n")
                if line.strip() == "```":
                    # Closing fence
                    self._emit_code_block(self._code_buf.rstrip("\n"), self._code_lang)
                    self._code_buf = ""
                    self._code_lang = ""
                    self._state = _StreamState.PROSE
                else:
                    self._code_buf += self._line_buf
                self._line_buf = ""

    def _refresh_prose(self, include_partial: bool = False) -> None:
        """Update the prose widget with accumulated lines."""
        if self._prose_widget:
            lines = list(self._prose_lines)
            if include_partial and self._line_buf:
                lines.append(self._line_buf)
            self._prose_widget.set_content("\n".join(lines))
            self._refresh_layout()

    def _emit_code_block(self, code: str, language: str) -> None:
        """Mount a CodeBlock widget for completed fenced code."""
        if not code.strip():
            return
        block = CodeBlock(code, language=language or "text", provider=self._provider)
        try:
            target = self._body_column or self
            target.mount(block)
        except Exception:
            pass
        # Start a new prose widget after the code block
        self._prose_lines = []
        self._prose_widget = _ProseBody("")
        try:
            target = self._body_column or self
            target.mount(self._prose_widget)
        except Exception:
            pass
        self._refresh_layout()

    def _refresh_layout(self) -> None:
        self.refresh(layout=True)
        if self.parent is not None:
            self.parent.refresh(layout=True)


class _ProseBody(Static):
    """A Static widget that re-renders its content when updated."""

    DEFAULT_CSS = """
    _ProseBody {
        width: 1fr;
        height: auto;
    }
    """

    def __init__(self, content: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._content = content

    def set_content(self, content: str) -> None:
        self._content = content
        self.refresh(layout=True)

    def render(self) -> Text:
        if not self._content:
            return Text("")
        return render_content(self._content)
