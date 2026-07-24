"""Codex-style tool-activity compaction.

A run of consecutive tool calls collapses into one bounded, self-scrolling
box (ToolActivityLog) with short rows, instead of pages of stacked full-width
rows. File writes/edits and failed calls break out and stay standalone/visible.
"""

import io

import pytest
from rich.console import Console
from textual.app import App, ComposeResult

from cascade.widgets.message import ChatHistory
from cascade.widgets.diff_block import WriteBlock
from cascade.widgets.tool_call import (
    ToolActivityLog,
    ToolCallWidget,
    _ToolActivityRow,
    append_tool_activity,
)


class _Harness(App):
    def compose(self) -> ComposeResult:
        yield ChatHistory()


def _rows(log: ToolActivityLog) -> list[_ToolActivityRow]:
    return [c for c in log.children if isinstance(c, _ToolActivityRow)]


def _content(chat: ChatHistory) -> list:
    return [c for c in chat.children if isinstance(c, (ToolActivityLog, WriteBlock, ToolCallWidget))]


@pytest.mark.asyncio
async def test_consecutive_plain_tools_fold_into_one_log():
    app = _Harness()
    async with app.run_test() as pilot:
        chat = app.query_one(ChatHistory)
        await pilot.pause()
        for i in range(5):
            append_tool_activity(chat, "read_file", {"path": f"src/f{i}.py"}, "content")
            await pilot.pause()
        content = _content(chat)
        assert len(content) == 1
        log = content[0]
        assert isinstance(log, ToolActivityLog)
        assert len(_rows(log)) == 5


@pytest.mark.asyncio
async def test_log_is_bounded_and_scrolls_within_itself():
    app = _Harness()
    async with app.run_test() as pilot:
        chat = app.query_one(ChatHistory)
        await pilot.pause()
        for i in range(20):
            append_tool_activity(chat, "list_files", {"path": f"d{i}"}, "ok")
            await pilot.pause()
        log = _content(chat)[0]
        # Bounded: the box does not grow past its max-height (10 cells)...
        assert log.size.height <= 10
        # ...and 20 rows overflow it, so it scrolls WITHIN itself.
        assert log.max_scroll_y > 0


@pytest.mark.asyncio
async def test_write_file_breaks_out_and_starts_new_log():
    app = _Harness()
    async with app.run_test() as pilot:
        chat = app.query_one(ChatHistory)
        await pilot.pause()
        append_tool_activity(chat, "read_file", {"path": "a.py"}, "x")
        await pilot.pause()
        append_tool_activity(chat, "read_file", {"path": "b.py"}, "y")
        await pilot.pause()
        append_tool_activity(chat, "write_file", {"path": "c.py", "content": "z=1\n"}, "wrote c.py")
        await pilot.pause()
        append_tool_activity(chat, "read_file", {"path": "d.py"}, "w")
        await pilot.pause()
        content = _content(chat)
        assert [type(c).__name__ for c in content] == [
            "ToolActivityLog", "WriteBlock", "ToolActivityLog",
        ]
        assert len(_rows(content[0])) == 2  # a.py, b.py
        assert len(_rows(content[2])) == 1  # d.py (fresh log after the write)


@pytest.mark.asyncio
async def test_failed_call_stays_visible_standalone():
    app = _Harness()
    async with app.run_test() as pilot:
        chat = app.query_one(ChatHistory)
        await pilot.pause()
        append_tool_activity(chat, "read_file", {"path": "ok.py"}, "fine")
        await pilot.pause()
        append_tool_activity(chat, "read_file", {"path": "bad.py"}, "Error: permission denied")
        await pilot.pause()
        content = _content(chat)
        # The error is NOT folded into the log -- it is a standalone, visible row.
        assert [type(c).__name__ for c in content] == ["ToolActivityLog", "ToolCallWidget"]
        err = content[1]
        console = Console(width=80, file=io.StringIO())
        console.print(err.query_one("_ToolBody").render())
        assert "permission denied" in console.file.getvalue()


def test_row_render_is_compact_no_result_blob():
    blob = "LINE_ONE_MARKER\n" + "x" * 400
    row = _ToolActivityRow("read_file", {"path": "src/module.py"}, blob)
    console = Console(width=80, file=io.StringIO())
    console.print(row.render())
    out = console.file.getvalue()
    assert "read_file" in out
    assert "src/module.py" in out
    assert "ok" in out
    # The full raw result must NOT be dumped inline -- that is the screen flood.
    assert "xxxx" not in out
    assert "LINE_ONE_MARKER" not in out


def test_row_render_surfaces_error():
    row = _ToolActivityRow("run_command", {"command": "pytest"}, "Error: 3 failed")
    console = Console(width=120, file=io.StringIO())
    console.print(row.render())
    out = console.file.getvalue()
    assert "run_command" in out
    assert "pytest" in out
    assert "3 failed" in out


def test_key_arg_prefers_command_over_first_value():
    from cascade.widgets.tool_call import _key_arg

    assert _key_arg({"command": "ls -la"}) == "ls -la"
    assert _key_arg({"pattern": "TODO", "path": "src"}) == "src"  # path wins by priority
    assert _key_arg({"weird": "value"}) == "value"  # fallback to first string
    assert _key_arg({}) == ""


@pytest.mark.asyncio
async def test_tool_call_widget_still_renders():
    # Regression guard: the standalone one-liner (touched module) still composes.
    app = _Harness()
    async with app.run_test() as pilot:
        chat = app.query_one(ChatHistory)
        await pilot.pause()
        w = ToolCallWidget("read_file", {"path": "x.py"}, "content")
        chat.mount(w)
        await pilot.pause()
        assert w.is_mounted
        console = Console(width=80, file=io.StringIO())
        console.print(w.query_one("_ToolBody").render())
        assert "read_file" in console.file.getvalue()
