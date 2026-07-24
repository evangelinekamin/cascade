"""Compact widget showing a completed tool call in the chat history."""

from pathlib import PurePosixPath

from rich.text import Text
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Static

from ..theme import PALETTE

# Tools whose calls read better as a rendered file block than a one-liner.
_WRITE_TOOLS = ("write_file", "append_file")
_EDIT_TOOLS = ("replace_in_file",)

# Map common file extensions to Rich Syntax lexer names.
_LEXERS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".jsx": "jsx", ".rs": "rust", ".go": "go", ".java": "java", ".c": "c",
    ".cpp": "cpp", ".h": "c", ".rb": "ruby", ".sh": "bash", ".css": "css",
    ".html": "html", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".md": "markdown", ".sql": "sql",
}


def _lexer_for(path: str) -> str:
    return _LEXERS.get(PurePosixPath(path).suffix.lower(), "text")


def _is_failure(tool_output: str) -> bool:
    """A tool result signals failure so it is never buried in a collapsed box."""
    return tool_output.strip().lower().startswith(("error", "no change"))


def render_tool_widget(tool_name: str, tool_input: dict, tool_output: str):
    """Pick the richest widget for a completed tool call.

    File writes render as a WriteBlock (syntax-highlighted content), edits
    as a DiffBlock (old vs new), everything else as the compact one-liner.
    A failed call always falls back to the one-liner so the error shows.
    """
    from .diff_block import DiffBlock, WriteBlock

    failed = _is_failure(tool_output)
    path = tool_input.get("path") or tool_input.get("file_path") or ""

    if not failed and tool_name in _WRITE_TOOLS and isinstance(path, str) and path:
        content = tool_input.get("content", "")
        if isinstance(content, str):
            return WriteBlock(path, content, language=_lexer_for(path))

    if not failed and tool_name in _EDIT_TOOLS and isinstance(path, str) and path:
        old = tool_input.get("old_string", "")
        new = tool_input.get("new_string", "")
        if isinstance(old, str) and isinstance(new, str):
            diff_lines = _line_diff(old, new)
            changed = sum(1 for _, op, _ in diff_lines if op in ("+", "-"))
            return DiffBlock(path, diff_lines, lines_changed=changed)

    return ToolCallWidget(tool_name, tool_input, tool_output)


def _line_diff(old: str, new: str) -> list[tuple[int, str, str]]:
    """A simple line-level diff for a surgical edit, using difflib."""
    import difflib

    old_lines = old.splitlines() or [""]
    new_lines = new.splitlines() or [""]
    out: list[tuple[int, str, str]] = []
    ln = 1
    for line in difflib.Differ().compare(old_lines, new_lines):
        tag, text = line[:2], line[2:]
        if tag == "- ":
            out.append((ln, "-", text))
        elif tag == "+ ":
            out.append((ln, "+", text))
            ln += 1
        elif tag == "  ":
            out.append((ln, " ", text))
            ln += 1
        # "? " hint lines are skipped
    return out


class ToolCallWidget(Widget):
    """A single tool-call row: gutter label + tool body."""

    DEFAULT_CSS = """
    ToolCallWidget {
        height: auto;
        width: 100%;
        padding: 0 0 0 0;
        layout: horizontal;
    }
    """

    def __init__(
        self,
        tool_name: str,
        tool_input: dict,
        tool_output: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._tool_input = tool_input
        self._tool_output = tool_output

    def compose(self) -> ComposeResult:
        yield _ToolGutter()
        yield _ToolBody(self._tool_name, self._tool_input, self._tool_output)


class _ToolGutter(Static):
    """Fixed-width gutter showing 'tool' in dim text."""

    DEFAULT_CSS = """
    _ToolGutter {
        width: 10;
        min-width: 10;
        max-width: 10;
        height: auto;
        text-align: right;
        padding-right: 1;
    }
    """

    def render(self) -> Text:
        return Text(f"{'tool':>8}", style=f"dim {PALETTE.text_dim}")


class _ToolBody(Static):
    """Tool name + truncated args + truncated result."""

    DEFAULT_CSS = """
    _ToolBody {
        width: 1fr;
        height: auto;
        padding-left: 1;
    }
    """

    def __init__(
        self, tool_name: str, tool_input: dict, tool_output: str, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._tool_input = tool_input
        self._tool_output = tool_output

    def render(self) -> Text:
        t = Text()
        t.append(self._tool_name, style=f"bold {PALETTE.file_ops}")

        # Truncated args
        args_str = ""
        if self._tool_input:
            import json
            try:
                args_str = json.dumps(self._tool_input, ensure_ascii=False)
            except Exception:
                args_str = str(self._tool_input)
        if args_str:
            if len(args_str) > 80:
                args_str = args_str[:77] + "..."
            t.append(f" {args_str}", style=f"dim {PALETTE.text_dim}")

        # Truncated result
        result = self._tool_output.strip()
        if result:
            if len(result) > 120:
                result = result[:117] + "..."
            result_oneline = result.replace("\n", " ")
            t.append(" -> ", style=f"dim {PALETTE.text_dim}")
            t.append(result_oneline, style=PALETTE.text_dim)

        return t


# ---------------------------------------------------------------------------
# Codex-style tool-activity compaction
#
# A run of consecutive tool calls collapses into a single bounded box that
# scrolls WITHIN itself, so a 30-call turn occupies a small fixed region
# instead of pages of transcript. Full results already live in the model
# context and on disk, so each row is intentionally just name + key arg +
# short status. File writes/edits and failed calls break out of the box so a
# change is never hidden and an error is never buried.
# ---------------------------------------------------------------------------

# Arg keys ordered by how well they identify a call, most salient first.
_ARG_KEYS = ("path", "file_path", "command", "cmd", "pattern", "query", "url")

# Trailing widgets that are not settled content, skipped when deciding whether
# the previous row is an open activity log to continue.
_TRANSIENT = frozenset(
    {"ThinkingIndicator", "StreamMessage", "_ProseBody",
     "AutocompleteDropdown", "OverflowIndicator"}
)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _key_arg(tool_input: dict) -> str:
    """The single most identifying argument of a call (path, command, ...)."""
    if not isinstance(tool_input, dict):
        return ""
    for key in _ARG_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return _clip(value, 56)
    for value in tool_input.values():
        if isinstance(value, str) and value:
            return _clip(value, 56)
    return ""


def _status(tool_output: str, failed: bool) -> str:
    """A terse status: the error's first line, or a plain 'ok' on success."""
    stripped = tool_output.strip()
    if not failed:
        return "ok"
    first = stripped.splitlines()[0] if stripped else ""
    return _clip(first, 48) or "error"


class _ToolActivityRow(Static):
    """One collapsed tool call: name + key arg + short status, single line."""

    DEFAULT_CSS = """
    _ToolActivityRow {
        height: 1;
        width: 100%;
    }
    """

    def __init__(
        self, tool_name: str, tool_input: dict, tool_output: str, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._tool_input = tool_input
        self._tool_output = tool_output

    def render(self) -> Text:
        t = Text(no_wrap=True, overflow="ellipsis")
        t.append(self._tool_name, style=f"bold {PALETTE.file_ops}")
        arg = _key_arg(self._tool_input)
        if arg:
            t.append(f"  {arg}", style=PALETTE.text_dim)
        failed = _is_failure(self._tool_output)
        t.append(f"  {_status(self._tool_output, failed)}",
                 style=PALETTE.error if failed else PALETTE.text_muted)
        return t


class ToolActivityLog(VerticalScroll):
    """Bounded, self-scrolling box collapsing a run of tool calls.

    Height is capped (max-height in CSS) and overflow scrolls internally, so
    the newest call stays in view while older rows recede without pushing the
    conversation off screen. Rows queued before mount are flushed on mount so
    the first call in a run is never dropped.
    """

    DEFAULT_CSS = f"""
    ToolActivityLog {{
        height: auto;
        max-height: 10;
        width: 100%;
        margin: 1 0;
        padding: 0 1;
        border: round {PALETTE.border};
        background: {PALETTE.code_bg};
        overflow-y: auto;
        overflow-x: hidden;
    }}
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pending: list[_ToolActivityRow] = []
        self._count = 0

    def add_row(self, tool_name: str, tool_input: dict, tool_output: str) -> None:
        row = _ToolActivityRow(tool_name, tool_input, tool_output)
        self._count += 1
        self.border_title = self._title()
        if self.is_mounted:
            self.mount(row)
            self.scroll_end(animate=False)
        else:
            self._pending.append(row)

    def on_mount(self) -> None:
        self.border_title = self._title()
        if self._pending:
            self.mount(*self._pending)
            self._pending = []
            self.scroll_end(animate=False)

    def _title(self) -> Text:
        label = "tool" if self._count == 1 else "tools"
        return Text(f" {self._count} {label} ", style=f"dim {PALETTE.text_dim}")


def _trailing_log(chat: Widget) -> ToolActivityLog | None:
    """The open activity log to continue, or None if the run was broken.

    A settled write/edit block, message, or standalone error since the last
    log ends the run, so the next plain call opens a fresh box.
    """
    for child in reversed(list(chat.children)):
        if type(child).__name__ in _TRANSIENT:
            continue
        return child if isinstance(child, ToolActivityLog) else None
    return None


def append_tool_activity(
    chat: Widget, tool_name: str, tool_input: dict, tool_output: str,
) -> Widget:
    """Mount a completed tool call, collapsing successful plain calls into a
    single ToolActivityLog while keeping file writes/edits and failures as
    standalone, always-visible blocks. Returns the owning widget.
    """
    widget = render_tool_widget(tool_name, tool_input, tool_output)
    if isinstance(widget, ToolCallWidget) and not _is_failure(tool_output):
        log = _trailing_log(chat)
        if log is None:
            log = ToolActivityLog()
            chat.mount(log)
        log.add_row(tool_name, tool_input, tool_output)
        return log
    chat.mount(widget)
    return widget
