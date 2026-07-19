"""Inline diff block for file write/edit operations.

Write block: amber accent, "> write path" header, full code.
Edit block: amber accent, "* edit path" header, red/green diff.
"""

from rich.text import Text
from rich.panel import Panel
from textual.widgets import Static

from ..theme import PALETTE

# A file write/edit should be visually distinct, not bury the conversation.
# Show a head window and a "N more lines" footer -- the full content is on
# disk (the tool wrote it) and in the tool result.
_MAX_BODY_LINES = 14


class DiffBlock(Static):
    """Inline diff viewer for file changes."""

    DEFAULT_CSS = """
    DiffBlock {
        height: auto;
        width: 100%;
        margin: 1 0;
    }
    """

    def __init__(
        self,
        file_path: str,
        diff_lines: list[tuple[int, str, str]],
        lines_changed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._file_path = file_path
        self._diff_lines = diff_lines
        self._lines_changed = lines_changed

    def render(self) -> Panel:
        title = Text()
        title.append(" \u2727 edit ", style=f"bold {PALETTE.amber}")
        title.append(self._file_path, style=PALETTE.text_primary)

        subtitle = Text(f" {self._lines_changed} lines changed ", style=f"dim {PALETTE.text_dim}")

        # Fold to a window centered on the changed hunks so a large edit does
        # not dump the whole file; changed lines are what the reader wants.
        shown = self._diff_lines
        hidden = 0
        if len(shown) > _MAX_BODY_LINES:
            changed_idx = [i for i, (_, op, _) in enumerate(shown) if op in ("+", "-")]
            if changed_idx:
                lo = max(0, changed_idx[0] - 2)
                hi = min(len(shown), changed_idx[-1] + 3)
                if hi - lo > _MAX_BODY_LINES:
                    hi = lo + _MAX_BODY_LINES
            else:
                lo, hi = 0, _MAX_BODY_LINES
            hidden = len(shown) - (hi - lo)
            shown = shown[lo:hi]

        content = Text()
        for ln, op, line in shown:
            if op == "-":
                content.append(f"{ln:>3}- ", style=f"bold {PALETTE.diff_del}")
                content.append(f"{line}\n", style=f"strikethrough {PALETTE.diff_del}")
            elif op == "+":
                content.append(f"{ln:>3}+ ", style=f"bold {PALETTE.diff_add}")
                content.append(f"{line}\n", style=f"bold {PALETTE.diff_add}")
            else:
                content.append(f"{ln:>3}  ", style=f"dim {PALETTE.text_dim}")
                content.append(f"{line}\n", style=PALETTE.text_primary)
        if hidden > 0:
            content.append(f"     … {hidden} more lines", style=f"dim {PALETTE.text_dim}")

        return Panel(
            content,
            title=title,
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style=f"{PALETTE.amber} 40%",
            background=PALETTE.code_bg,
            padding=(0, 1),
            expand=True,
        )


class WriteBlock(Static):
    """Block showing a full file write (new file creation)."""

    DEFAULT_CSS = """
    WriteBlock {
        height: auto;
        width: 100%;
        margin: 1 0;
    }
    """

    def __init__(self, file_path: str, code: str, language: str = "text", **kwargs) -> None:
        super().__init__(**kwargs)
        self._file_path = file_path
        self._code = code
        self._language = language

    def render(self) -> Panel:
        from rich.syntax import Syntax

        title = Text()
        title.append(" > write ", style=f"bold {PALETTE.amber}")
        title.append(self._file_path, style=PALETTE.text_primary)

        lines = self._code.split("\n")
        total = len(lines)
        if total > _MAX_BODY_LINES:
            code = "\n".join(lines[:_MAX_BODY_LINES])
            subtitle = Text(
                f" {total} lines · showing {_MAX_BODY_LINES} ",
                style=f"dim {PALETTE.text_dim}",
            )
        else:
            code = self._code
            subtitle = Text(f" {total} lines ", style=f"dim {PALETTE.text_dim}")

        syntax = Syntax(
            code,
            self._language,
            theme="monokai",
            line_numbers=True,
            word_wrap=False,
            background_color=PALETTE.code_bg,
        )

        return Panel(
            syntax,
            title=title,
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style=f"{PALETTE.amber} 40%",
            background=PALETTE.code_bg,
            padding=(0, 1),
            expand=True,
        )
