"""render_tool_widget: file writes/edits become rich blocks, not one-liners."""

from cascade.widgets.tool_call import ToolCallWidget, render_tool_widget, _line_diff
from cascade.widgets.diff_block import DiffBlock, WriteBlock


class TestFactory:
    def test_write_file_becomes_write_block(self):
        w = render_tool_widget(
            "write_file", {"path": "src/module.py", "content": "x = 1\n"}, "wrote src/module.py",
        )
        assert isinstance(w, WriteBlock)
        assert w._language == "python"

    def test_append_file_becomes_write_block(self):
        w = render_tool_widget("append_file", {"path": "notes.md", "content": "hi"}, "ok")
        assert isinstance(w, WriteBlock)
        assert w._language == "markdown"

    def test_replace_becomes_diff_block(self):
        w = render_tool_widget(
            "replace_in_file",
            {"path": "a.py", "old_string": "a=1\nb=2", "new_string": "a=1\nb=3"},
            "ok",
        )
        assert isinstance(w, DiffBlock)
        assert w._lines_changed == 2  # one - and one +

    def test_run_command_stays_oneliner(self):
        w = render_tool_widget("run_command", {"command": "ls"}, "files")
        assert isinstance(w, ToolCallWidget)

    def test_read_file_stays_oneliner(self):
        w = render_tool_widget("read_file", {"path": "x.py"}, "content")
        assert isinstance(w, ToolCallWidget)

    def test_failed_write_falls_back_to_oneliner_so_error_shows(self):
        w = render_tool_widget("write_file", {"path": "x", "content": "y"}, "Error: denied")
        assert isinstance(w, ToolCallWidget)
        w2 = render_tool_widget(
            "replace_in_file", {"path": "x", "old_string": "a", "new_string": "b"},
            "No change to x: old_string not found",
        )
        assert isinstance(w2, ToolCallWidget)

    def test_missing_path_falls_back(self):
        w = render_tool_widget("write_file", {"content": "orphan"}, "ok")
        assert isinstance(w, ToolCallWidget)


class TestLineDiff:
    def test_marks_removals_and_additions(self):
        diff = _line_diff("old line\nkeep", "new line\nkeep")
        ops = [op for _, op, _ in diff]
        assert "-" in ops and "+" in ops and " " in ops

    def test_pure_addition(self):
        diff = _line_diff("", "brand new")
        assert any(op == "+" for _, op, _ in diff)
