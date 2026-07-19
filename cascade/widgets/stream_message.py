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

from .message import render_content, render_md_line, GutterLabel, GutterSeparator
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
        # How many completed prose lines have been handed to the widget and
        # frozen; only lines past this and the partial re-render per batch.
        self._flushed_lines = 0
        self._prose_widget: _ProseBody | None = None
        self._body_column: Vertical | None = None

    def compose(self) -> ComposeResult:
        yield GutterLabel(self._provider)
        yield GutterSeparator(self._provider)
        with Vertical(classes="stream-body"):
            self._prose_widget = _ProseBody("")
            yield self._prose_widget

    def on_mount(self) -> None:
        try:
            self._body_column = self.query_one(".stream-body", Vertical)
        except Exception:
            self._body_column = None

    def feed(self, chunk: str) -> None:
        """Feed a streaming chunk. Handles arbitrary chunk boundaries.

        Rendering refreshes once per fed batch, not once per completed line: the
        caller coalesces chunks on a ~30ms cadence, so a single refresh here keeps
        the stream smooth instead of forcing a full re-layout on every newline.
        """
        for ch in chunk:
            self._process_char(ch)
        if self._state == _StreamState.PROSE:
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
                    # Accumulate the completed line; the refresh is coalesced to
                    # once per fed batch in feed() to avoid per-line layout churn.
                    self._prose_lines.append(line)
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
        """Push freshly-completed lines to the widget, plus the live partial.

        Completed lines are rendered ONCE and frozen by the widget; only the
        trailing partial re-renders per batch. This keeps long streams O(n)
        overall instead of the O(n^2) of re-rendering the whole segment on
        every 30ms batch.
        """
        if not self._prose_widget:
            return
        # Hand over any prose lines completed since the last refresh.
        if len(self._prose_lines) > self._flushed_lines:
            new_lines = self._prose_lines[self._flushed_lines:]
            self._prose_widget.append_lines(new_lines)
            self._flushed_lines = len(self._prose_lines)
        partial = self._line_buf if include_partial else ""
        self._prose_widget.set_partial(partial)
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
        self._flushed_lines = 0
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
    """Prose widget that freezes completed lines and re-renders only the tail.

    ``append_lines`` renders each completed line exactly once into a frozen
    Text buffer; ``set_partial`` swaps the trailing in-progress line. render()
    is then frozen + partial, so a growing stream costs one line-render per
    completed line rather than re-parsing the whole segment every batch.
    """

    DEFAULT_CSS = """
    _ProseBody {
        width: 1fr;
        height: auto;
    }
    """

    def __init__(self, content: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._frozen = Text()
        self._partial = ""
        self._frozen_line_count = 0
        if content:
            self.append_lines(content.split("\n"))

    def append_lines(self, lines: list[str]) -> None:
        """Render and freeze newly-completed lines (each rendered once)."""
        for line in lines:
            if self._frozen_line_count > 0:
                self._frozen.append("\n")
            self._frozen.append_text(render_md_line(line))
            self._frozen_line_count += 1
        self.refresh(layout=True)

    def set_partial(self, partial: str) -> None:
        """Set the trailing in-progress line (the only part that re-renders)."""
        if partial == self._partial:
            return
        self._partial = partial
        self.refresh(layout=True)

    def render(self) -> Text:
        if self._frozen_line_count == 0 and not self._partial:
            return Text("")
        if not self._partial:
            return self._frozen.copy()
        out = self._frozen.copy()
        if self._frozen_line_count > 0:
            out.append("\n")
        out.append_text(render_md_line(self._partial))
        return out
