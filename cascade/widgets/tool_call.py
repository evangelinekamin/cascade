"""Compact widget showing a completed tool call in the chat history."""

from pathlib import PurePosixPath

from rich.text import Text
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


def render_tool_widget(tool_name: str, tool_input: dict, tool_output: str):
    """Pick the richest widget for a completed tool call.

    File writes render as a WriteBlock (syntax-highlighted content), edits
    as a DiffBlock (old vs new), everything else as the compact one-liner.
    A failed call always falls back to the one-liner so the error shows.
    """
    from .diff_block import DiffBlock, WriteBlock

    failed = tool_output.strip().lower().startswith(("error", "no change"))
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
